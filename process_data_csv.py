import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm  

# Define target image size
IMAGE_SHAPE = (52, 208, 1)  # Height=52, Width=208, 1-channel (grayscale)

# Load CSVs
# df_train = pd.read_csv("dataset/written_name_train_v2.csv")
# df_test = pd.read_csv("dataset/written_name_test_v2.csv")
df_val = pd.read_csv("dataset/written_name_validation_v2.csv")

# Construct file paths
# train_images = ["dataset/train_v2/train/" + fname for fname in df_train["FILENAME"]]
# test_images = ["dataset/test_v2/test/" + fname for fname in df_test["FILENAME"]]
val_images = ["dataset/validation_v2/validation/" + fname for fname in df_val["FILENAME"]]

# train_labels = df_train["IDENTITY"].astype(str).tolist()
# test_labels = df_test/["IDENTITY"].astype(str).tolist()
val_labels = df_val["IDENTITY"].astype(str).tolist()

# train_images = train_images[:len(train_images)//2]
# train_labels = train_labels[:len(train_labels)//2]
val_images = val_images[:len(val_images)//2]
val_labels = val_labels[:len(val_labels)//2]
# test_images = test_images[:len(test_images)//2]
# test_labels = test_labels[:len(test_labels)//2]
# Image preprocessing function
def load_image(image_path, target_size=(52, 208)):
    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img = img.resize(target_size)  
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return img_array.reshape(52, 208, 1)  # Ensure shape is (52,208,1)
    except Exception as e:
        print(f"⚠️ Error loading {image_path}: {e}")
        return None  

# Function to process images & labels
def process_csv_in_batches(image_paths, labels, output_csv, batch_size=10000):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(labels)  # Fit tokenizer once
    
    with open(output_csv, "w") as f:
        f.write("images,labels\n")  # Write header
        
    for i in range(0, len(image_paths), batch_size):
        batch_images = []
        batch_labels = []
        
        for img_path, label in zip(image_paths[i:i+batch_size], labels[i:i+batch_size]):
            img_array = load_image(img_path)
            if img_array is not None:
                batch_images.append(",".join(map(str, img_array.flatten())))  # Flatten image
                batch_labels.append(",".join(map(str, tokenizer.texts_to_sequences([label])[0])))  # Encode label
        
        df_batch = pd.DataFrame({"images": batch_images, "labels": batch_labels})
        df_batch.to_csv(output_csv, mode="a", header=False, index=False)  # Append to CSV
        
        print(f"✅ Processed {i + batch_size}/{len(image_paths)}")

# Call function for each dataset
# process_csv_in_batches(train_images, train_labels, "train_process_data.csv")
process_csv_in_batches(val_images, val_labels, "val_process_data.csv")
# process_csv_in_batches(test_images, test_labels, "test_process_data.csv")
