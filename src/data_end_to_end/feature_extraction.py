import cv2
import mediapipe as mp
import numpy as np
import csv
import time

gesture_name="neutral_raw"#just need to change this to choose the directory
save_dir = "/Users/anoopchhabra/Documents/College/Projects/handgesture/data/raw/gestures_raw/"+gesture_name+"/"

mp_hand=mp.solutions.hands
hand=mp_hand.Hands(max_num_hands=1)
mp_drawing=mp.solutions.drawing_utils

cap=cv2.VideoCapture(1)
frame_count=0

while True:
    succ,frame=cap.read()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    extracted=hand.process(rgb)
    if extracted.multi_hand_landmarks:
        for hands in extracted.multi_hand_landmarks:
            coordinates=[]
            for extracted in hands.landmark:
                coordinates.append(extracted.x)
                coordinates.append(extracted.y)
                coordinates.append(extracted.z)
            mp_drawing.draw_landmarks(
                frame, hands, mp_hand.HAND_CONNECTIONS)
    file=open(save_dir+f"{frame_count}"+".csv", "w+") 
    writer= csv.writer(file)
    try:
        writer.writerow(coordinates)
    except NameError:
        time.sleep(1)
        continue
    frame_count+=1
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()