import cv2
import time
import os
import mediapipe as mp
from utils import HandDetector 


cap = cv2.VideoCapture("https://192.168.1.106:8080/video")

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)

print("Captured Width: ", width)
print("Captured Height: ", height)
print("Captured FPS: ", fps)

# Create video writer object to write video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('/home/dev2/Documents/mediapipe_hand_detection/output.avi',fourcc, int(fps), (width,height))

start_time = 0#  time.time()

hand_detector = HandDetector(detection_confidence=0.75)
tip_id = [4, 8, 12, 16, 20]

while True:
    ret, img = cap.read()
    if not ret: 
        break
    img = hand_detector.get_hands(img)
    hand_landmark = hand_detector.hand_position(img)

    if len(hand_landmark) != 0:
        fingers = []

        # Thumb Index of 0
        if hand_landmark[tip_id[0]][1] > hand_landmark[tip_id[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four Fingers Index 1 to 5 
        for id in range(1, 5):
            if hand_landmark[tip_id[id]][2] < hand_landmark[tip_id[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        total_fingers = fingers.count(1)
        
        cv2.rectangle(img, (25, 215), (295, 325), (0, 255, 0), cv2.FILLED)
        if total_fingers == 0:
            cv2.putText(img, "  Status: Closed", (25, 245), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1.1, (0, 0, 0), 2)    
        elif total_fingers == 5:
            cv2.putText(img, "  Status: Opened", (25, 245), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1.1, (0, 0, 0), 2) 


        cv2.rectangle(img, (25, 250), (295, 290), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f" No Of Fingers", (25, 275), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 2)
        cv2.putText(img, f"{str(total_fingers)}", (150, 325), cv2.FONT_HERSHEY_PLAIN,
                    2.5, (0, 0, 0), 3)
        out.write(img)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    start_time = end_time

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
