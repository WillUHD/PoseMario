import cv2, numpy as np, time, threading, pyautogui, mediapipe as mp, os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

pyautogui.PAUSE = 0 
modelFile       = "mediaPipePose.task"
webcam          = 0
conf            = 0.8

holdDuration   = 0.5
cooldown        = 0.15
jumpJolt  = 250   
jumpThresh     = -0.3  
duckThresh      = -jumpThresh

cRun       = (0, 255, 120)
cSprint    = (200, 100, 255)
cBack      = (255, 100, 50)
cJump      = (0, 200, 255)
cDuck      = (255, 200, 0)
cIdle      = (40, 40, 40)
cText      = (100, 100, 100)
cWhite           = (255, 255, 255)

bounding = [
    (-1.8, -1.0, -1.0, 1.0, "SPRINT BACK"),
    (-1.0, -0.5, -1.0, 1.0, "BACK"),
    (0.5,  1.2,  -1.0, 1.0, "RUN"),
    (1.2,  2.5,  -1.0, 1.0, "SPRINT"),
]

detResult       = None
detTimestamp    = -1
resultLock      = threading.Lock()

def resultCallback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detResult, detTimestamp
    with resultLock: 
        detResult = result
        detTimestamp = timestamp_ms

def dRectTransparent(img, pt1, pt2, color, alpha):
    x1, y1 = max(0, min(pt1[0], pt2[0])), max(0, min(pt1[1], pt2[1]))
    x2, y2 = min(img.shape[1], max(pt1[0], pt2[0])), min(img.shape[0], max(pt1[1], pt2[1]))
    if x1 >= x2 or y1 >= y2: return
    roi = img[y1:y2, x1:x2]
    rect = np.full_like(roi, color)
    cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0, roi)

def dJoints(img, detection_result):
    if not detection_result or not detection_result.pose_landmarks: return img
    for joint in detection_result.pose_landmarks:
        jointProto = landmark_pb2.NormalizedLandmarkList()
        jointProto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in joint
        ])
        solutions.drawing_utils.draw_landmarks(
            img, jointProto, solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return img

if __name__ == "__main__":
    modelPath = os.path.join(os.path.dirname(__file__), modelFile)
    if not os.path.exists(modelPath): modelPath = modelFile
    
    baseOpt = python.BaseOptions(model_asset_path=modelPath)
    opts = vision.PoseLandmarkerOptions(
        base_options=baseOpt,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=conf,
        min_tracking_confidence=conf,
        result_callback=resultCallback
    )

    try: landmarker = vision.PoseLandmarker.create_from_options(opts)
    except Exception as e:
        print(f"Init Error: {e}"); exit()

    watermarkImg = cv2.imread("./watermark.png", cv2.IMREAD_UNCHANGED)
    cap = cv2.VideoCapture(webcam)
    
    inferenceFPS = 15 
    inferenceDelay = 1.0 / inferenceFPS
    lastInfTime = 0
    lastProcessed = -1
    
    y_history = deque() 
    v_expiry = 0 
    current_v_state = 'NEUTRAL'
    current_h_state = 'NEUTRAL'

    paused = False
    pressedKeys = set()
    t = time.time()
    wmCache = None

    # Pre-fetch Landmark Indices
    L_SH = solutions.pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
    L_WR = solutions.pose.PoseLandmark.LEFT_WRIST.value
    R_WR = solutions.pose.PoseLandmark.RIGHT_WRIST.value

    while True:
        ret, frame = cap.read()
        if not ret: break

        now = time.time()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'):
            paused = not paused
            if paused:
                for k in list(pressedKeys): pyautogui.keyUp(k)
                pressedKeys.clear(); v_expiry = 0
            else: y_history.clear()

        if paused:
            cv2.putText(frame, "PAUSED", (10, 140), 1, 2, (0, 255, 255), 2)
            cv2.imshow('PoseMario - willuhd - MediaPipe', frame)
            continue

        if now - lastInfTime >= inferenceDelay:
            work_frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mpImage, int(now * 1000))
            lastInfTime = now
        
        frame = cv2.flip(frame, 1)

        with resultLock: res, ts = detResult, detTimestamp

        valid_pose = False
        if res and res.pose_landmarks:
            pose = res.pose_landmarks[0]
            h, w = frame.shape[:2]
            
            # Visibility Check
            if pose[L_SH].visibility > conf and pose[R_SH].visibility > conf:
                p_l = np.array([pose[L_SH].x * w, pose[L_SH].y * h])
                p_r = np.array([pose[R_SH].x * w, pose[R_SH].y * h])
                s = np.linalg.norm(p_r - p_l) # Shoulder width in pixels

                if s > 50:
                    valid_pose = True
                    mid = (p_l + p_r) * 0.5
                    
                    # 2. Update Input Logic (Sync with Inference)
                    if ts != lastProcessed:
                        lastProcessed = ts
                        
                        y_history.append((ts, mid[1]))
                        while y_history and (ts - y_history[0][0]) > jumpJolt:
                            y_history.popleft()
                        
                        # Trigger Check: Only if not already in an action or cooldown
                        if now > v_expiry + cooldown:
                            if len(y_history) > 1:
                                delta_y = mid[1] - y_history[0][1]
                                if delta_y < jumpThresh * s:
                                    v_expiry = now + holdDuration
                                    current_v_state = 'JUMP'
                                    y_history.clear() # Prevent rebound
                                elif delta_y > duckThresh * s:
                                    v_expiry = now + holdDuration
                                    current_v_state = 'DUCK'
                                    y_history.clear() # Prevent rebound
                        
                        # Determine current vertical state based on timer
                        if now >= v_expiry:
                            current_v_state = 'NEUTRAL'

                        # Horizontal Logic (Wrist detection)
                        current_h_state = 'NEUTRAL'
                        wrists = [pose[L_WR], pose[R_WR]]
                        for wrist in (w for w in wrists if w.visibility > conf):
                            nx = ((wrist.x * w) - mid[0]) / s
                            if nx < -0.5:
                                current_h_state = 'BACK_SPRINT' if nx < -1.0 else 'BACK'
                            elif nx > 0.5:
                                current_h_state = 'SPRINT' if nx > 1.2 else 'RUN'
                        
                        if current_v_state == 'DUCK': current_h_state = 'NEUTRAL'

                        # 3. Input Dispatch
                        triggeredKeys = set()
                        if current_h_state == 'SPRINT': triggeredKeys.update(['d', 'r']) 
                        elif current_h_state == 'RUN': triggeredKeys.add('d')
                        elif current_h_state == 'BACK_SPRINT': triggeredKeys.update(['a', 'r']) 
                        elif current_h_state == 'BACK': triggeredKeys.add('a')

                        if current_v_state == 'DUCK': triggeredKeys.update(['s', 'r'])
                        elif current_v_state == 'JUMP': triggeredKeys.add('space')
                            
                        for k in (triggeredKeys - pressedKeys): pyautogui.keyDown(k)
                        for k in (pressedKeys - triggeredKeys): pyautogui.keyUp(k)
                        pressedKeys = triggeredKeys

                    # 4. Render UI
                    for x1, x2, y1, y2, lbl in bounding:
                        pt1 = (int(mid[0] + x1 * s), int(mid[1] + y1 * s))
                        pt2 = (int(mid[0] + x2 * s), int(mid[1] + y2 * s))
                        dRectTransparent(frame, pt1, pt2, cIdle, 0.3)
                        cv2.putText(frame, lbl, (pt1[0]+5, pt2[1]-5), 1, 0.5, cText, 1)
                    
                    label, color, ax1, ax2 = "", None, -0.5, 0.5
                    if current_h_state == 'SPRINT': ax1, ax2, color, label = 0.5, 2.5, cSprint, "SPRINT"
                    elif current_h_state == 'RUN': ax1, ax2, color, label = 0.5, 1.2, cRun, "RUN"
                    elif current_h_state == 'BACK_SPRINT': ax1, ax2, color, label = -1.8, -0.5, cSprint, "SPRINT BACK"
                    elif current_h_state == 'BACK': ax1, ax2, color, label = -1.0, -0.5, cBack, "BACK"
                    
                    if current_v_state == 'DUCK': color, label = cDuck, "DUCK"
                    elif current_v_state == 'JUMP': color, label = cJump, "JUMP"

                    if color:
                        p1 = (int(mid[0] + ax1 * s), int(mid[1] + -1.0 * s))
                        p2 = (int(mid[0] + ax2 * s), int(mid[1] + 1.0 * s))
                        dRectTransparent(frame, p1, p2, color, 0.4)
                        cv2.rectangle(frame, p1, p2, cWhite, 2)
                        cv2.putText(frame, label, (p1[0], p1[1]-10), 1, 1.5, cWhite, 2)
                
            frame = dJoints(frame, res)

        if not valid_pose:
            y_history.clear()
            v_expiry = 0
            if pressedKeys:
                for k in list(pressedKeys): pyautogui.keyUp(k)
                pressedKeys.clear()

        # Watermark Overlay (Optimized blending)
        if watermarkImg is not None:
            if wmCache is None:
                fH, fW = frame.shape[:2]
                origH, origW = watermarkImg.shape[:2]
                scale = fW / origW
                nw, nh = int(origW * scale), int(origH * scale)
                resized = cv2.resize(watermarkImg, (nw, nh))
                x, y = (fW - nw) // 2, max(0, fH - nh)
                if resized.shape[2] == 4:
                    mask = (resized[:, :, 3] / 255.0)[:, :, np.newaxis]
                    fg = (resized[:, :, :3] * mask).astype(np.uint8)
                    wmCache = (fg, 1.0 - mask, x, y, nw, nh)
                else: wmCache = (resized, None, x, y, nw, nh)
            
            fg, inv_mask, x, y, nw, nh = wmCache
            roi = frame[y:y+nh, x:x+nw]
            if inv_mask is not None:
                roi[:] = fg + (roi * inv_mask).astype(np.uint8)
            else: roi[:] = fg
        
        # Performance FPS Counter
        curr_t = time.time()
        fps = 1 / (curr_t - t) if (curr_t - t) > 0 else 0
        t = curr_t
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 1, 1.5, (0, 255, 0), 2)
        cv2.imshow('PoseMario - willuhd - MediaPipe', frame)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()