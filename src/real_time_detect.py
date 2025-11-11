import csv
import pickle as pkl
import numpy as np
from nn.model import NeuralNetwork
from nn.optimizers_functions import Adam
from nn.activation_function import Softmax,Tanh
from nn.loss import cross_entropy,cross_entropy_diff
from nn.dense import Dense
from nn.model import NeuralNetwork
import pandas as pd
import mediapipe as mp
import cv2

loaded_nn = NeuralNetwork.load("/Users/anoopchhabra/Documents/College/Projects/handgesture/models/gesture_model.pkl")

mp_hand=mp.solutions.hands
hand=mp_hand.Hands(max_num_hands=1)
mp_drawing=mp.solutions.drawing_utils

cap=cv2.VideoCapture(1)
frame_count=0

label={}
file=open("/Users/anoopchhabra/Documents/College/Projects/handgesture/src/label.csv")
read=csv.reader(file)
list_label=list(read)
for i in list_label:
    label[i[1]]=i[0]

while True:
    #Obtaining the data
    succ,frame=cap.read()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    extracted=hand.process(rgb)
    try:
        if extracted.multi_hand_landmarks:
            for hands in extracted.multi_hand_landmarks:
                coordinates=[]
                for extracted in hands.landmark:
                    coordinates.append(extracted.x)
                    coordinates.append(extracted.y)
                    coordinates.append(extracted.z)
                mp_drawing.draw_landmarks(
                    frame, hands, mp_hand.HAND_CONNECTIONS)
        #Normalizing it
        landmarks = np.array(coordinates).reshape(-1, 3)
        wrist = landmarks[0]
        landmarks -= wrist
        scale = np.max(np.abs(landmarks))
        if scale!=0:
            landmarks /= scale
        coordinates=landmarks.flatten().tolist()
    #Predicting
    except NameError:
        coordinates=np.zeros((1,63))
    a = loaded_nn.predict(np.array(coordinates).reshape(1, -1))
    c=np.argmax(a[0])
    if a[0][c]>.7:
        print(label[str(int(c))])
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()