import cv2, numpy as np, time, threading, pyautogui, mediapipe as mp, os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model 			= "mediaPipePose.task"
webcam 			= 0
confidence 		= 0.8
detResult 		= None
resultLock 		= threading.Lock()
frameTime 		= 0
lastTime		= 0

runColor    	= (0, 255, 120)
sprintColor 	= (200, 100, 255)
backColor   	= (255, 100, 50)
jumpColor   	= (255, 200, 0)
idleColor   	= (40, 40, 40)
textColor   	= (100, 100, 100)
white 			= (255, 255, 255)
green 			= (0, 255, 0)

backSprintEnd 	= -3.0
backSprintStart	= -1.6
backStart 		= -0.8
runStart		= 0.8
sprintStart		= 1.6
sprintEnd		= 3.0

jumpEnd			= -2.0
jumpStart    	= -1.0
zoneHeightLimit = 0.8

rects = [
	(backSprintEnd, 	backSprintStart, jumpStart,	zoneHeightLimit,	"SPRINT BACK"),
	(backSprintStart,	backStart,		 jumpStart,	zoneHeightLimit,	"BACK"),
	(runStart,			sprintStart,	 jumpStart,	zoneHeightLimit,	"RUN"),
	(sprintStart,		sprintEnd,		 jumpStart,	zoneHeightLimit,	"SPRINT"),
	(backStart,			runStart,		 jumpEnd,	jumpStart,			"JUMP"),
]

minShoulderWidth	= 50
leftShoulder 		= solutions.pose.PoseLandmark.LEFT_SHOULDER.value
rightShoulder 		= solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
leftWrist 			= solutions.pose.PoseLandmark.LEFT_WRIST.value
rightWrist 			= solutions.pose.PoseLandmark.RIGHT_WRIST.value
leftHip             = solutions.pose.PoseLandmark.LEFT_HIP.value
rightHip            = solutions.pose.PoseLandmark.RIGHT_HIP.value

def resultCallback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	global detResult
	with resultLock: detResult = result

def drawJoints(rgb_image, detection_result, active_idx=0):
	if not detection_result or not detection_result.pose_landmarks: return rgb_image
	if active_idx < len(detection_result.pose_landmarks):
		joint = detection_result.pose_landmarks[active_idx]
		overlay = np.copy(rgb_image)
		jointProto = landmark_pb2.NormalizedLandmarkList()
		jointProto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in joint
		])
		solutions.drawing_utils.draw_landmarks(
			overlay,
			jointProto,
			solutions.pose.POSE_CONNECTIONS,
			solutions.drawing_styles.get_default_pose_landmarks_style())
		return overlay
	return rgb_image

def get_h_state(nx):
	if nx < backStart:
		if nx < backSprintStart: return 'BACK_SPRINT'
		else: return 'BACK'
	elif nx > runStart:
		if nx < sprintStart: return 'RUN'
		else: return 'SPRINT'
	return 'NEUTRAL'

if __name__ == "__main__":
	modelPath = os.path.join(os.path.dirname(__file__), model)
	if not os.path.exists(modelPath): modelPath = model
	baseOpt = python.BaseOptions(model_asset_path=modelPath)

	opts = vision.PoseLandmarkerOptions(
		base_options					= baseOpt,
		running_mode					= vision.RunningMode.LIVE_STREAM,
		num_poses						= 5,
		min_pose_detection_confidence	= confidence,
		min_tracking_confidence			= confidence,
		result_callback					= resultCallback
	)

	try: landmarker = vision.PoseLandmarker.create_from_options(opts)
	except Exception as e:
		print(f"Failed to create PoseLandmarker {e}")
		exit()

	watermarkImg = cv2.imread("./watermark.png", cv2.IMREAD_UNCHANGED)

	cap = cv2.VideoCapture(webcam)
	if not cap.isOpened(): raise IOError("Cannot open webcam")

	paused = False
	pressedKeys = set()
	t = time.time()
	wmCache = None
	lastFrameDims = None

	last_left_zone = 'NEUTRAL'
	last_right_zone = 'NEUTRAL'
	primary_arm = None

	# Jump Logic Variables
	jumpActive = False
	lastJumpChange = 0
	jumpHoldTime = 0.3  # How long to hold space (seconds)
	jumpCoolDown = 0.1  # How long to wait before re-pressing (seconds)

	while True:
		if not paused:
			ret, frame 	= cap.read()
			if not ret: break
			frame 		= cv2.flip(frame, 1)
			rgbFrame 	= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			mpImage		= mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
			
			frameTime 	= int(time.time() * 1000)
			if frameTime <= lastTime: frameTime = lastTime + 1
			lastTime 	= frameTime
			
			landmarker.detect_async(mpImage, frameTime)

			current_result = None
			with resultLock: 
				if detResult: current_result = detResult

			pyautogui.PAUSE = 0
			
			active_pose_idx = -1

			if current_result and current_result.pose_landmarks:
				best_dist = 1.0
				for i, pl in enumerate(current_result.pose_landmarks):
					hx = (pl[leftHip].x + pl[rightHip].x) * 0.5
					dist = abs(hx - 0.5)
					if dist < best_dist:
						best_dist = dist
						active_pose_idx = i
				
				if active_pose_idx != -1:
					pose_landmarks	= current_result.pose_landmarks[active_pose_idx]
					h, w, _ 		= frame.shape
					shoulder_l 		= pose_landmarks[leftShoulder]
					shoulder_r 		= pose_landmarks[rightShoulder]
					
					shoulders_visible = (shoulder_l.visibility > confidence and
										 shoulder_r.visibility > confidence)

					if shoulders_visible:
						p_sh_l = np.array([shoulder_l.x * w, shoulder_l.y * h])
						p_sh_r = np.array([shoulder_r.x * w, shoulder_r.y * h])
						
						width_px = np.linalg.norm(p_sh_r - p_sh_l)

						if width_px > minShoulderWidth:
							mid = (p_sh_l + p_sh_r) * 0.5
							s = width_px 
							overlay = frame.copy()
							
							current_h_state = 'NEUTRAL'
							current_v_state = 'NEUTRAL'
							
							wrist_l_pt = pose_landmarks[leftWrist]
							wrist_r_pt = pose_landmarks[rightWrist]

							left_zone = 'NEUTRAL'
							if wrist_l_pt.visibility > confidence:
								wx, wy = wrist_l_pt.x * w, wrist_l_pt.y * h
								nx, ny = (wx - mid[0]) / s, (wy - mid[1]) / s
								if ny < jumpStart: current_v_state = 'JUMP'
								elif ny > zoneHeightLimit: left_zone = 'NEUTRAL'
								else: left_zone = get_h_state(nx)

							right_zone = 'NEUTRAL'
							if wrist_r_pt.visibility > confidence:
								wx, wy = wrist_r_pt.x * w, wrist_r_pt.y * h
								nx, ny = (wx - mid[0]) / s, (wy - mid[1]) / s
								if ny < jumpStart: current_v_state = 'JUMP'
								elif ny > zoneHeightLimit: right_zone = 'NEUTRAL'
								else: right_zone = get_h_state(nx)

							left_activated = (left_zone != 'NEUTRAL' and last_left_zone == 'NEUTRAL')
							right_activated = (right_zone != 'NEUTRAL' and last_right_zone == 'NEUTRAL')

							if left_activated: primary_arm = 'LEFT'
							if right_activated: primary_arm = 'RIGHT'

							if primary_arm == 'LEFT' and left_zone == 'NEUTRAL':
								if right_zone != 'NEUTRAL': primary_arm = 'RIGHT'
								else: primary_arm = None
							elif primary_arm == 'RIGHT' and right_zone == 'NEUTRAL':
								if left_zone != 'NEUTRAL': primary_arm = 'LEFT'
								else: primary_arm = None
							
							if primary_arm is None:
								if left_zone != 'NEUTRAL': primary_arm = 'LEFT'
								elif right_zone != 'NEUTRAL': primary_arm = 'RIGHT'

							if primary_arm == 'LEFT': current_h_state = left_zone
							elif primary_arm == 'RIGHT': current_h_state = right_zone
							else: current_h_state = 'NEUTRAL'

							last_left_zone = left_zone
							last_right_zone = right_zone

							for x1, x2, y1, y2, lbl in rects:
								pt1 = (int(mid[0] + x1 * s), int(mid[1] + y1 * s))
								pt2 = (int(mid[0] + x2 * s), int(mid[1] + y2 * s))
								cv2.rectangle(overlay, pt1, pt2, idleColor, -1)
								cx, cy = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
								cv2.putText(overlay, lbl, (cx - 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, cv2.LINE_AA)
							
							label = ""
							color = None
							ax1, ax2, ay1, ay2 = -0.8, 0.8, jumpStart, zoneHeightLimit
							triggeredKeys = set()
							
							if current_h_state == 'SPRINT':
								ax1, ax2 = runStart, sprintEnd
								color = sprintColor
								label = "SPRINT"
								triggeredKeys.update(['d', 'r']) 
							elif current_h_state == 'RUN':
								ax1, ax2 = runStart, sprintStart
								color = runColor
								label = "RUN"
								triggeredKeys.add('d')
							elif current_h_state == 'BACK_SPRINT':
								ax1, ax2 = backSprintEnd, backStart
								color = sprintColor
								label = "SPRINT BACK"
								triggeredKeys.update(['a', 'r']) 
							elif current_h_state == 'BACK':
								ax1, ax2 = backSprintStart, backStart
								color = backColor
								label = "BACK"
								triggeredKeys.add('a')

							if current_v_state == 'JUMP':
								ay1, ay2 = jumpEnd, jumpStart
								label = "JUMP" if label == "" else label + " + JUMP"
								if color is None: color = jumpColor
								
								# Time Match Jump Logic
								currT = time.time()
								if not jumpActive:
									if currT - lastJumpChange > jumpCoolDown:
										pyautogui.keyDown('space')
										jumpActive = True
										lastJumpChange = currT
								else:
									if currT - lastJumpChange > jumpHoldTime:
										pyautogui.keyUp('space')
										jumpActive = False
										lastJumpChange = currT
							else:
								if jumpActive:
									pyautogui.keyUp('space')
									jumpActive = False
									lastJumpChange = time.time()
								
							toPress = triggeredKeys - pressedKeys
							toRelease = pressedKeys - triggeredKeys
							for k in toPress: pyautogui.keyDown(k)
							for k in toRelease: pyautogui.keyUp(k)
							pressedKeys = triggeredKeys
							
							if color is not None:
								pt1 = (int(mid[0] + ax1 * s), int(mid[1] + ay1 * s))
								pt2 = (int(mid[0] + ax2 * s), int(mid[1] + ay2 * s))
								cv2.rectangle(overlay, pt1, pt2, color, -1)
								cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
								cv2.rectangle(frame, pt1, pt2, white, 2)
								ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
								cx, cy = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
								tx, ty = cx - ts[0] // 2, cy + ts[1] // 2
								cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
							else: cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

						else: cv2.putText(frame, "Invalid pose", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				
				if active_pose_idx != -1:
					frame = drawJoints(frame, current_result, active_pose_idx)

			if watermarkImg is not None:
				fH, fW = frame.shape[:2]
				if wmCache is None or lastFrameDims != (fW, fH):
					origH, origW = watermarkImg.shape[:2]
					scale = 1.0
					if origW > fW: scale = fW / origW
					
					finalW = int(origW * scale)
					finalH = int(origH * scale)
					
					resized = watermarkImg
					if scale != 1.0: resized = cv2.resize(watermarkImg, (finalW, finalH), interpolation=cv2.INTER_AREA)
					xPos = (fW - finalW) // 2
					yPos = fH - finalH
					if yPos < 0:
						resized = resized[-yPos:, :, :]
						finalH = fH
						yPos = 0

					if resized.shape[2] == 4:
						alpha = resized[:, :, 3] / 255.0
						alpha = np.expand_dims(alpha, axis=2) 
						invAlpha = 1.0 - alpha
						fg = resized[:, :, :3] * alpha
						wmCache = (resized[:, :, :3], fg, invAlpha, xPos, yPos)
					else: wmCache = (resized, None, None, xPos, yPos)
					
					lastFrameDims = (fW, fH)

				visImg, fg, invAlpha, xPos, yPos = wmCache
				h_wm, w_wm = visImg.shape[:2]
				roi = frame[yPos:yPos+h_wm, xPos:xPos+w_wm]

				if fg is not None:
					blended = (fg + roi * invAlpha)
					roi[:] = blended.astype(np.uint8)
				else: roi[:] = visImg
		
		current_time = time.time()
		fps = 1 / (current_time - t) if (current_time - t) > 0 else 0
		t = current_time
		
		if paused: cv2.putText(frame, "Paused: P to resume in 3s", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
		else: cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

		cv2.imshow('PoseMario - willuhd - MediaPipe', frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'): break
		elif key == ord('p'):
			paused = not paused
			if paused:
				for k in list(pressedKeys): pyautogui.keyUp(k)
				pressedKeys.clear()
				if jumpActive:
					pyautogui.keyUp('space')
					jumpActive = False
			else: time.sleep(3)

	landmarker.close()
	for k in list(pressedKeys):
		pyautogui.keyUp(k)
	if jumpActive: pyautogui.keyUp('space')
	cap.release()
	cv2.destroyAllWindows()