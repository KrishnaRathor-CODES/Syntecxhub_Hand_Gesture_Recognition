import os
import cv2
import math
import urllib.request
import mediapipe as mp
import pyautogui

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

NEON_GREEN = (100, 255, 100)
NEON_PINK = (200, 100, 255)
NEON_YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

TIP_IDS = [4, 8, 12, 16, 20]
BASE_IDS = [2, 6, 10, 14, 18]

def get_fingers_state(pts):
    wrist = pts[0]
    fingers = []
    for tip, base in zip(TIP_IDS, BASE_IDS):
        tip_dist = math.hypot(pts[tip][0] - wrist[0], pts[tip][1] - wrist[1])
        base_dist = math.hypot(pts[base][0] - wrist[0], pts[base][1] - wrist[1])
        fingers.append(1 if tip_dist > base_dist * 1.15 else 0)
    return fingers

def classify_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    if fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Peace Sign"
    return None

GESTURE_ACTIONS = {
    "Open Palm": "Play / Pause",
    "Fist": "Mute",
    "Thumbs Up": "Volume Up",
    "Peace Sign": "Volume Down",
}

GESTURE_KEYS = {
    "Open Palm": "playpause",
    "Fist": "volumemute",
    "Thumbs Up": "volumeup",
    "Peace Sign": "volumedown",
}

def trigger_system_action(gesture):
    key = GESTURE_KEYS.get(gesture)
    if key:
        try:
            pyautogui.press(key)
        except Exception:
            pass

def draw_skeleton(frame, lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], NEON_GREEN, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, NEON_YELLOW, -1, cv2.LINE_AA)
    return pts

def draw_hud(frame, gesture, action, w, h):
    cv2.rectangle(frame, (0, 0), (w, 45), (30, 30, 30), -1)
    cv2.putText(frame, "Hand Gesture Recognition", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_YELLOW, 2)

    gesture_text = gesture if gesture else "No gesture detected"
    action_text = action if action else "-"

    cv2.rectangle(frame, (0, h - 55), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, f"Gesture: {gesture_text}", (15, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_GREEN, 2)
    cv2.putText(frame, f"Action: {action_text}", (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_PINK, 2)

def main():
    ensure_model()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_gesture = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        gesture = None

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            pts = draw_skeleton(frame, lm, w, h)
            fingers = get_fingers_state(pts)
            gesture = classify_gesture(fingers)

        if gesture and gesture != last_gesture:
            trigger_system_action(gesture)
        last_gesture = gesture

        current_action = GESTURE_ACTIONS.get(gesture) if gesture else None

        draw_hud(frame, gesture, current_action, w, h)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
