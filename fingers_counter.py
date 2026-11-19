import cv2
import mediapipe as mp
import urllib.request
import time, math
import os, pyautogui, keyboard

# 1. Download the required model bundle (run once)
#model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
#urllib.request.urlretrieve(model_url, "hand_landmarker.task")



#set up the tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FINGERTIP_IDS1 = [0, 4, 8, 12, 16, 20]
FINGERTIP_IDS = [0,1, 2, 3, 4, 5, 6, 7, 8,9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20]

# Create the landmarker in VIDEO mode for synchronous webcam processing
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 3,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)
lock_x = [0,0]
lock_y = [0,0]


def main():

    pyautogui.PAUSE = 0

    trigerd = False
    frame_count = 0

    is_pinching = False
    located = 0
    axe = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        sum = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_image = mp.Image(image_format= mp.ImageFormat.SRGB, data = rgb_frame)

        timespam_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timespam_ms)

        if result.hand_landmarks:
            last_seen_time = time.time()
            for hand_landmarks, hand_info in zip(result.hand_landmarks, result.handedness):
                
                for tip_id in FINGERTIP_IDS:
                    landmark = hand_landmarks[tip_id]

                    cx, cy = int(landmark.x * w), int(landmark.y *h)

                    if(tip_id in FINGERTIP_IDS1 ):
                        landmark2 = hand_landmarks[tip_id - 2]
                        cx2, cy2 = int(landmark2.x * w), int(landmark2.y * h)

                        if(tip_id == 4):
                            landmark2 = hand_landmarks[tip_id - 1]
                            cx2, cy2 = int(landmark2.x * w), int(landmark2.y * h)
                            if cx < cx2:
                                sum += 1
                        else:
                            if cy < cy2:
                                sum += 1

                if trigerd and (time.time() > (time.time() - frame_count)> 0.5) and (sum == 0):
                    keyboard.send("play/pause media")
                    trigerd = False
                    frame_count = 0
                    print(sum)
                               
                if(sum == 5 and (not trigerd)):
                    frame_count = time.time()
                    trigerd = True
                    print("trigred")
                    
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                x4, y4 = int(thumb.x * w), int(thumb.y * h)
                x8, y8 = int(index.x * w), int(index.y * h)
                distance = math.hypot(x8 - x4, y8 - y4)

                if not is_pinching:
                    if sum == 4 and distance < 25:
                        pyautogui.moveTo(0.5*w, 0.5*h)
                        is_pinching = True

            # 2. Once pinching, IGNORE `sum` — only release when fingers separate
                else:
                    if distance > 45:
                        is_pinching = False
                        located = 0
                        pyautogui.mouseUp(button='left')
                        pyautogui.moveRel(0, -40)

                if is_pinching:
                    if located == 0:
                        try:
                            anchor = pyautogui.locateCenterOnScreen("assets/volume.png", confidence=0.8)
                            if anchor:
                                timeline_x = anchor.x - 80
                                timeline_y = anchor.y - 50
                                pyautogui.moveTo(timeline_x, timeline_y)
                                pyautogui.mouseDown(button='left')
                                axe = x4
                                located = 1
                        except Exception:
                            located = 3
                            print("image not found")

                    elif located == 1:
                        dx = x4 - axe
                        if abs(dx) > 3:
                            pyautogui.moveRel(dx * 5, 0)
                            axe = x4

                

                    

                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), cv2.FILLED)

                    cv2.putText(
                    frame, 
                    f"ID:{tip_id}", 
                    (cx - 15, cy - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.25, 
                    (0, 255, 0), 
                    1
                    )
        else:
            # 4. Grace period: only release if hand disappears for > 0.25 seconds
            if is_pinching and (time.time() - last_seen_time) > 0.25:
                is_pinching = False
                located = 0
                pyautogui.mouseUp(button='left')
                pyautogui.moveRel(0, -40)
        #frame[0:200, 0:200] = overlay_list[sum]
        #cv2.putText(frame, f"{sum}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 1)
        #cv2.imshow("webCam", frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

if __name__ == "__main__":
    main()