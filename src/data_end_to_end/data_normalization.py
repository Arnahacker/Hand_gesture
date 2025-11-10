import csv
import numpy as np
import pandas as pd
import os

raw_path = "/Users/anoopchhabra/Documents/College/Projects/handgesture/data/raw/gestures_raw/thumbs_up_raw"
normalized_dir = "/Users/anoopchhabra/Documents/College/Projects/handgesture/data/processed/gesture_processed/thumbs_up"
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    scale = np.max(np.abs(landmarks))
    landmarks /= scale
    return landmarks.flatten().tolist()

all_data = []

for file in sorted(os.listdir(raw_path)):
    if file.endswith(".csv"):
        with open(os.path.join(raw_path, file), 'r') as f:
            reader = csv.reader(f)
            row = next(reader)
            row = [float(x) for x in row]
            normalized = normalize_landmarks(row)
            all_data.append(normalized)

df = pd.DataFrame(all_data)
df.to_csv(os.path.join(normalized_dir, "normalized_closed_fist.csv"), index=False)
