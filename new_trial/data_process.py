import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

# Define dataset paths
train_csv_path = "../dataset/written_name_train_v2.csv"
test_csv_path = "../dataset/written_name_test_v2.csv"
val_csv_path = "../dataset/written_name_validation_v2.csv"

train_image_dir = "../dataset/train_v2/train/"
test_image_dir = "../dataset/test_v2/test/"
val_image_dir = "../dataset/validation_v2/validation/"

# Define image size
IMG_WIDTH = 208
IMG_HEIGHT = 52

# Function to load and preprocess an image
def load_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img = img.resize(target_size)  # Resize image
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize (0-1)
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None  # Return None if image is not found

# Function to process CSV and save as new CSV with image arrays
def process_csv(csv_path, image_dir, output_csv):
    df = pd.read_csv(csv_path)
    image_paths = image_dir + df["FILENAME"]
    labels = df["IDENTITY"]

    processed_data = []

    for img_path, label in tqdm(zip(image_paths, labels), total=len(df), desc=f"Processing {output_csv}"):
        img_array = load_image(img_path)
        if img_array is not None:
            processed_data.append([img_path, label, img_array.flatten().tolist()])  # Flatten the array

    # Convert to DataFrame and save
    df_processed = pd.DataFrame(processed_data, columns=["image_path", "label", "image_array"])
    df_processed.to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_csv}")

# Process train, validation, and test datasets
process_csv(train_csv_path, train_image_dir, "train_processed.csv")
process_csv(val_csv_path, val_image_dir, "val_processed.csv")
process_csv(test_csv_path, test_image_dir, "test_processed.csv")
