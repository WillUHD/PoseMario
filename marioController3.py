import cv2, numpy as np, time, threading, pyautogui, mediapipe as mp, os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration & Constants ---
pyautogui.PAUSE = 0 
MODEL_FILE      = "pose_landmarker_full.task"  # adjust lite/full/heavy based on hardware
WEBCAM_INDEX    = 0
CONFIDENCE      = 0.7
INFERENCE_WIDTH = 480

# Gameplay Logic
HOLD_DURATION   = 0.5
COOLDOWN        = 0.15
JUMP_JOLT       = 250
JUMP_THRESH     = -0.3  
DUCK_THRESH     = 0.3

# UI Styling
C_RUN, C_SPRINT, C_BACK = (0, 255, 120), (200, 100, 255), (255, 100, 50)
C_JUMP, C_DUCK, C_IDLE  = (0, 200, 255), (255, 200, 0), (40, 40, 40)
C_WHITE, C_TEXT         = (255, 255, 255), (200, 200, 200)

BOUNDING_BOXES = [
    (-1.8, -1.0, -1.0, 0.5, "SPRINT BACK"),
    (-1.0, -0.4, -1.0, 0.5, "BACK"),
    (0.4,  1.2,  -1.0, 0.5, "RUN"),
    (1.2,  2.5,  -1.0, 0.5, "SPRINT"),
]

# --- Global State & Async Callback ---
det_result = None
det_timestamp = -1
result_lock = threading.Lock()

def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global det_result, det_timestamp
    with result_lock:
        det_result = result
        det_timestamp = timestamp_ms

def draw_fast_rect(img, pt1, pt2, color, alpha):
    """Blends a transparent rectangle directly into the ROI."""
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x1 >= x2 or y1 >= y2: return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

# ============================================================
# NEW: Manual landmark drawing (replaces drawing_utils)
# ============================================================
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (27, 31), (28, 32), (29, 31), (30, 32)
]

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks manually using OpenCV."""
    if not detection_result or not detection_result.pose_landmarks:
        return
    
    h, w = rgb_image.shape[:2]
    
    for pose_landmarks in detection_result.pose_landmarks:
        # Extract landmark coordinates
        landmarks = []
        for lm in pose_landmarks:
            landmarks.append((int(lm.x * w), int(lm.y * h)))
        
        # Draw connections
        for connection in POSE_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                cv2.line(rgb_image, landmarks[idx1], landmarks[idx2], (0, 255, 0), 2)
        
        # Draw landmark points
        for lm in landmarks:
            cv2.circle(rgb_image, lm, 3, (0, 0, 255), -1)

# ============================================================

if __name__ == "__main__":
    # ============================================================
    # NEW: Initialize PoseLandmarker with Tasks API
    # ============================================================
    model_path = os.path.abspath(MODEL_FILE)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE,
        result_callback=result_callback
    )
    
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    watermark = cv2.imread("./watermark.png", cv2.IMREAD_UNCHANGED)
    
    # State Helpers
    inference_delay = 1.0 / 20 
    last_inf_time = 0
    last_processed_ts = -1
    y_history = deque() 
    v_expiry = 0
    cur_v, cur_h = 'NEUTRAL', 'NEUTRAL'
    pressed_keys = set()
    paused = False
    wm_cache = None
    fps_time = time.time()

    # Landmark indices (same as before)
    L_SH, R_SH = 11, 12
    L_WR, R_WR = 15, 16

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        now = time.time()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'):
            paused = not paused
            if paused:
                for k in list(pressed_keys): pyautogui.keyUp(k)
                pressed_keys.clear()
            continue

        if paused:
            cv2.putText(frame, "PAUSED", (w//2-100, h//2), 1, 3, (0, 255, 255), 3)
            cv2.imshow('PoseMario Optimized', frame)
            continue

        # 1. Async Inference (On Downscaled Frame)
        if now - last_inf_time >= inference_delay:
            small_h = int(h * (INFERENCE_WIDTH / w))
            small_frame = cv2.resize(frame, (INFERENCE_WIDTH, small_h))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(now * 1000))
            last_inf_time = now

        # 2. Logic & Control
        with result_lock:
            res, ts = det_result, det_timestamp

        valid_pose = False
        if res and res.pose_landmarks:
            pose = res.pose_landmarks[0]  # List of NormalizedLandmark objects
            
            # ============================================================
            # NEW: Access landmarks with .x, .y, .z (same as before)
            # ============================================================
            if pose[L_SH].visibility > CONFIDENCE and pose[R_SH].visibility > CONFIDENCE:
                p_l = np.array([pose[L_SH].x * w, pose[L_SH].y * h])
                p_r = np.array([pose[R_SH].x * w, pose[R_SH].y * h])
                scale = np.linalg.norm(p_r - p_l)
                mid = (p_l + p_r) * 0.5

                if scale > 40:
                    valid_pose = True
                    if ts != last_processed_ts:
                        last_processed_ts = ts
                        y_history.append((ts, mid[1]))
                        while y_history and (ts - y_history[0][0]) > JUMP_JOLT: y_history.popleft()

                        if now > v_expiry + COOLDOWN:
                            if len(y_history) > 1:
                                dy = mid[1] - y_history[0][1]
                                if dy < JUMP_THRESH * scale:
                                    v_expiry, cur_v = now + HOLD_DURATION, 'JUMP'
                                    y_history.clear()
                                elif dy > DUCK_THRESH * scale:
                                    v_expiry, cur_v = now + HOLD_DURATION, 'DUCK'
                                    y_history.clear()
                        
                        if now >= v_expiry: cur_v = 'NEUTRAL'

                        cur_h = 'NEUTRAL'
                        for wrist_idx in [L_WR, R_WR]:
                            wrist = pose[wrist_idx]
                            if wrist.visibility > CONFIDENCE:
                                nx, ny = (wrist.x * w - mid[0]) / scale, (wrist.y * h - mid[1]) / scale
                                for x1, x2, y1, y2, lbl in BOUNDING_BOXES:
                                    if x1 <= nx <= x2 and y1 <= ny <= y2: cur_h = lbl

                        new_keys = set()
                        if cur_v == 'JUMP': new_keys.add('space')
                        elif cur_v == 'DUCK': new_keys.update(['s', 'r'])
                        else:
                            if 'SPRINT' in cur_h: new_keys.add('r')
                            if 'BACK' in cur_h: new_keys.add('a')
                            elif 'RUN' in cur_h or 'SPRINT' in cur_h: new_keys.add('d')

                        for k in (new_keys - pressed_keys): pyautogui.keyDown(k)
                        for k in (pressed_keys - new_keys): pyautogui.keyUp(k)
                        pressed_keys = new_keys

                    # 3. UI Rendering (Full Resolution)
                    for x1, x2, y1, y2, lbl in BOUNDING_BOXES:
                        pt1 = (int(mid[0] + x1 * scale), int(mid[1] + y1 * scale))
                        pt2 = (int(mid[0] + x2 * scale), int(mid[1] + y2 * scale))
                        draw_fast_rect(frame, pt1, pt2, C_IDLE, 0.25)
                        cv2.putText(frame, lbl, (pt1[0], pt2[1]+15), 1, 0.8, C_TEXT, 1)
                    
                    act_lbl, act_col, coords = None, None, None
                    if cur_v != 'NEUTRAL':
                        act_lbl, act_col = cur_v, (C_JUMP if cur_v == 'JUMP' else C_DUCK)
                        coords = (-0.5, 0.5, -1.0, 0.5)
                    elif cur_h != 'NEUTRAL':
                        act_lbl, act_col = cur_h, (C_SPRINT if 'SPRINT' in cur_h else C_RUN if 'RUN' in cur_h else C_BACK)
                        coords = next((x1, x2, y1, y2) for x1, x2, y1, y2, lbl in BOUNDING_BOXES if lbl == cur_h)

                    if act_lbl and coords:
                        p1 = (int(mid[0] + coords[0] * scale), int(mid[1] + coords[2] * scale))
                        p2 = (int(mid[0] + coords[1] * scale), int(mid[1] + coords[3] * scale))
                        draw_fast_rect(frame, p1, p2, act_col, 0.4)
                        cv2.rectangle(frame, p1, p2, C_WHITE, 2)
                        cv2.putText(frame, act_lbl, (p1[0], p1[1]-10), 1, 1.5, C_WHITE, 2)

                    # ============================================================
                    # NEW: Manual drawing instead of drawing_utils
                    # ============================================================
                    draw_landmarks_on_image(frame, res)

        if not valid_pose:
            y_history.clear()
            v_expiry = 0
            if pressed_keys:
                for k in list(pressed_keys): pyautogui.keyUp(k)
                pressed_keys.clear()

        # 4. Watermark & FPS
        if watermark is not None:
            if wm_cache is None:
                scale_wm = w / watermark.shape[1]
                nw, nh = int(watermark.shape[1] * scale_wm), int(watermark.shape[0] * scale_wm)
                wm_res = cv2.resize(watermark, (nw, nh))
                mask = (wm_res[:, :, 3] / 255.0)[:, :, np.newaxis]
                wm_cache = ((wm_res[:, :, :3] * mask).astype(np.uint8), 1.0 - mask, 0, h - nh, nw, nh)
            
            fg, inv_m, wx, wy, wn, wh = wm_cache
            roi = frame[wy:wy+wh, wx:wx+wn]
            roi[:] = fg + (roi * inv_m).astype(np.uint8)

        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 1, 1.5, (0, 255, 0), 2)
        cv2.imshow('PoseMario Optimized', frame)

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()