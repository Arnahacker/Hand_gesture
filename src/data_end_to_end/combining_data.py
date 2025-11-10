import pandas as pd

gestures = {
    "closed_fist": "/Users/anoopchhabra/Documents/College/Projects/handgesture/data/processed/gesture_processed/closed_fist/normalized_closed_fist.csv",
    "thumbs_up": "/Users/anoopchhabra/Documents/College/Projects/handgesture/data/processed/gesture_processed/thumbs_up/normalized_closed_fist.csv",
    "open_hand": "/Users/anoopchhabra/Documents/College/Projects/handgesture/data/processed/gesture_processed/open_hand/normalized_closed_fist.csv",
    "thumbs_down":"/Users/anoopchhabra/Documents/College/Projects/handgesture/data/processed/gesture_processed/thumbs_down/normalized_closed_fist.csv",
    "index_up":"/Users/anoopchhabra/Documents/College/Projects/handgesture/data/processed/gesture_processed/index_up/normalized_closed_fist.csv"
}

all_data = []

for label, path in gestures.items():
    df = pd.read_csv(path)
    df["label"] = label
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("/Users/anoopchhabra/Documents/College/Projects/handgesture/data/label.csv", index=False)
