import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV file
df = pd.read_csv("dataset/written_name_train_v2.csv")

# Drop missing values
df.dropna(subset=["IDENTITY"], inplace=True)

# Shuffle data before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use only 50% of the dataset
df_half = df.iloc[:len(df) // 2]  # Select the first half

# Split into train (80%), validation (10%), and test (10%)
train_df, temp_df = train_test_split(df_half, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the split files
train_df.to_csv("dataset/train_half.csv", index=False)
val_df.to_csv("dataset/val_half.csv", index=False)
test_df.to_csv("dataset/test_half.csv", index=False)

print("Data split complete!")

import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

# Paths
image_dir = "dataset/train_v2/train/"
csv_files = {
    "train": "dataset/train_half.csv",
    "val": "dataset/val_half.csv",
    "test": "dataset/test_half.csv"
}

# Image size
image_shape = (64, 256)

# Load tokenizer
df_train = pd.read_csv(csv_files["train"])  # Use training data for tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df_train["IDENTITY"].astype(str))

# Function to process data
def process_data(csv_file, save_name):
    df = pd.read_csv(csv_file)
    image_data, labels = [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(image_dir, row["FILENAME"])
        
        try:
            img = Image.open(img_path).convert("L").resize(image_shape)
            img_array = np.array(img) / 255.0
            image_data.append(img_array)
            labels.append(tokenizer.texts_to_sequences([row["IDENTITY"]])[0])
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # Convert to NumPy and save
    np.save(f"{save_name}_images.npy", np.array(image_data))
    np.save(f"{save_name}_labels.npy", np.array(labels, dtype=object))

    print(f"Processed {save_name} data!")

# Process train, val, and test separately
process_data(csv_files["train"], "train")
process_data(csv_files["val"], "val_")
process_data(csv_files["test"], "test_")
