"""
Hand Gesture Mouse Controller for Slingshot / Angry Birds-style Games
=====================================================================

Prerequisites:
Install the required libraries before running this script:
    pip install opencv-python mediapipe pyautogui numpy

Controls:
- Move your hand inside the green bounding box on the screen to move the mouse cursor.
- Close your hand into a fist to press and HOLD the Right Mouse Button.
- Move your fist while closed to aim and pull back the slingshot.
- Open your hand to RELEASE the Right Mouse Button and launch!
- Move your mouse to any corner of your monitor to trigger PyAutoGUI's failsafe, 
  or press 'q' while the camera window is focused to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import math
import sys
import os
import urllib.request

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Frame Margin (in pixels): Defines the active region inside the camera frame.
# A larger margin means you have to move your hand less to cover the whole screen.
FRAME_MARGIN = 110

# Smoothing factor (1 = no smoothing/instant, higher = smoother but slightly more lag)
SMOOTHING = 4.5

# Which mouse button to hold ('left' for standard click/drag, 'right' for right-click)
MOUSE_BUTTON = 'left'

# MediaPipe Tasks Hand Landmarker model details
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Hand skeleton connection pairs for drawing landmarks with OpenCV
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

# Configure PyAutoGUI for real-time gaming responsiveness
pyautogui.PAUSE = 0  # Remove default 0.1s delay between PyAutoGUI calls
pyautogui.FAILSAFE = True  # Slam mouse to top-left corner of screen to emergency abort


def ensure_model_exists():
    """Downloads the MediaPipe Hand Landmarker model file if it isn't already in the folder."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading '{MODEL_PATH}' for MediaPipe Tasks API (first-time setup)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading model file: {e}")
            sys.exit(1)


def draw_hand_skeleton(frame, landmarks, frame_w, frame_h):
    """Draws hand connections and joints directly using OpenCV."""
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pts[start_idx], pts[end_idx], (200, 200, 200), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 255), cv2.FILLED)


def is_hand_closed(landmarks):
    """
    Determines whether the hand is closed (making a fist) or open.
    Calculates if the fingertips of the index, middle, ring, and pinky fingers
    are curled closer to the wrist than their respective middle knuckles (PIP joints).
    """
    wrist = landmarks[0]
    
    # MediaPipe landmark indices for: (Finger Tip, PIP Joint)
    finger_joints = [
        (8, 6),   # Index finger
        (12, 10), # Middle finger
        (16, 14), # Ring finger
        (20, 18)  # Pinky finger
    ]
    
    curled_fingers = 0
    for tip_idx, pip_idx in finger_joints:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        
        # Euclidean distance from the wrist (landmark 0)
        dist_tip_to_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_pip_to_wrist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        
        # If the tip is closer to the wrist than the middle joint, the finger is curled
        if dist_tip_to_wrist < dist_pip_to_wrist:
            curled_fingers += 1
            
    # Consider the hand a "fist" if at least 3 of the 4 main fingers are curled
    return curled_fingers >= 3


def main():
    # Ensure the required .task model file is downloaded
    ensure_model_exists()

    # Initialize the new MediaPipe Tasks HandLandmarker (VIDEO mode)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # Get the primary monitor resolution
    screen_w, screen_h = pyautogui.size()

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not access the webcam. Make sure it is connected and not used by another app.")
        sys.exit(1)

    # State variables for smoothing, timestamps, and click tracking
    prev_x, prev_y = screen_w // 2, screen_h // 2
    curr_x, curr_y = prev_x, prev_y
    is_clicking = False
    frame_timestamp_ms = 0

    print("====================================================")
    print(" Hand Gesture Controller Started!")
    print(f" Action Configured: HOLD & RELEASE '{MOUSE_BUTTON.upper()}' CLICK")
    print(" -> Open Hand + Move : Move Cursor")
    print(" -> Close Hand (Fist): Hold Right Click")
    print(" -> Move Closed Hand : Drag / Aim")
    print(" -> Open Hand Again  : Release Right Click (Launch!)")
    print(" -> Press 'q' in the video window to exit.")
    print("====================================================")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip frame horizontally for natural mirror-like hand movement
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape

            # Convert BGR image (OpenCV) to RGB and wrap in mp.Image for the Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # VIDEO mode requires a monotonically increasing timestamp in milliseconds
            frame_timestamp_ms += 33
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Draw the active control zone bounding box
            cv2.rectangle(
                frame,
                (FRAME_MARGIN, FRAME_MARGIN),
                (frame_w - FRAME_MARGIN, frame_h - FRAME_MARGIN),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Active Control Zone", (FRAME_MARGIN, FRAME_MARGIN - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                # Draw hand skeleton on the preview window
                draw_hand_skeleton(frame, hand_landmarks, frame_w, frame_h)

                # Use Landmark 9 (Middle Finger MCP / Knuckle) as the cursor anchor point.
                # Unlike fingertips, this central palm point remains completely stable 
                # when transitioning between an open hand and a closed fist!
                anchor = hand_landmarks[9]
                anchor_x = int(anchor.x * frame_w)
                anchor_y = int(anchor.y * frame_h)

                # Highlight the anchor tracking point on the video feed
                cv2.circle(frame, (anchor_x, anchor_y), 8, (255, 0, 255), cv2.FILLED)

                # Map the hand coordinates from the Active Control Zone to the full Screen Resolution
                target_x = np.interp(anchor_x, (FRAME_MARGIN, frame_w - FRAME_MARGIN), (0, screen_w))
                target_y = np.interp(anchor_y, (FRAME_MARGIN, frame_h - FRAME_MARGIN), (0, screen_h))

                # Apply exponential smoothing to reduce webcam jitter
                curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                curr_y = prev_y + (target_y - prev_y) / SMOOTHING

                # Clamp values strictly inside screen boundaries to avoid triggering corner failsafe accidentally
                safe_x = max(2, min(screen_w - 2, int(curr_x)))
                safe_y = max(2, min(screen_h - 2, int(curr_y)))

                # Move the mouse cursor
                pyautogui.moveTo(safe_x, safe_y)
                prev_x, prev_y = curr_x, curr_y

                # Check hand gesture state (Fist vs Open)
                fist_detected = is_hand_closed(hand_landmarks)

                if fist_detected:
                    # If hand just closed, press and hold the mouse button
                    if not is_clicking:
                        pyautogui.mouseDown(button=MOUSE_BUTTON)
                        is_clicking = True

                    # Visual feedback for charging/aiming
                    cv2.circle(frame, (anchor_x, anchor_y), 16, (0, 0, 255), 3)
                    cv2.putText(
                        frame, f"STATUS: HOLDING {MOUSE_BUTTON.upper()} (AIMING)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
                    )
                else:
                    # If hand just opened while holding click, release the mouse button
                    if is_clicking:
                        pyautogui.mouseUp(button=MOUSE_BUTTON)
                        is_clicking = False

                    # Visual feedback for open hand / released state
                    cv2.putText(
                        frame, "STATUS: OPEN HAND (RELEASED)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
                    )

            else:
                # Safety release: if the hand leaves the camera view while holding click, release it
                if is_clicking:
                    pyautogui.mouseUp(button=MOUSE_BUTTON)
                    is_clicking = False

                cv2.putText(
                    frame, "STATUS: NO HAND DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2
                )

            # Show the camera feed window
            cv2.imshow("Angry Birds Hand Controller (Press 'q' to quit)", frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except pyautogui.FailSafeException:
        print("\nPyAutoGUI FailSafe triggered (mouse moved to screen corner). Exiting safely...")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        # Ensure the mouse button isn't left stuck held down when closing the script
        if is_clicking:
            pyautogui.mouseUp(button=MOUSE_BUTTON)
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Controller stopped cleanly.")


if __name__ == "__main__":
    main()