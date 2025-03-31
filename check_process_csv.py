# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tqdm import tqdm

# # Load dataset
# df = pd.read_csv("dataset/written_name_train_v2.csv")
# df.dropna(subset=["IDENTITY"], inplace=True)

# # Define image directory
# image_dir = "dataset/train_v2/train/"
# batch_size = 50000  # Process in batches

# # Image processing function
# def load_and_process_image(image_path, target_size=(64, 256)):
#     try:
#         img = Image.open(image_path).convert("L")  # Convert to grayscale
#         img = img.resize(target_size)  # Resize
#         return np.array(img, dtype=np.float32) / 255.0  # Normalize
#     except Exception as e:
#         print(f"Error loading {image_path}: {e}")
#         return None

# # Initialize tokenizer
# tokenizer = Tokenizer(char_level=True)
# tokenizer.fit_on_texts(df["IDENTITY"].astype(str))

# # Create lists for all data
# all_images, all_labels = [], []

# # Process in batches
# for i in range(0, len(df), batch_size):
#     batch_df = df.iloc[i : i + batch_size]

#     image_data, labels = [], []

#     for _, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
#         img_path = os.path.join(image_dir, row["FILENAME"])
#         img_array = load_and_process_image(img_path)

#         if img_array is None:
#             continue  # Skip missing images

#         encoded_label = tokenizer.texts_to_sequences([row["IDENTITY"]])[0]

#         image_data.append(img_array)
#         labels.append(encoded_label)

#     all_images.extend(image_data)
#     all_labels.extend(labels)

# # Convert to NumPy arrays
# all_images = np.array(all_images)
# all_labels = np.array(all_labels, dtype=object)

# # Save everything into one file
# np.savez_compressed("train_data.npz", images=all_images, labels=all_labels)

# print("🚀 Processing complete! Data saved in 'train_data.npz'.")
import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

# 🔹 Load dataset
df = pd.read_csv("dataset/written_name_train_v2.csv")
df.dropna(subset=["IDENTITY"], inplace=True)

image_dir = "dataset/train_v2/train/"
batch_size = 50000 
image_shape = (64, 256) 
max_label_len = 20 

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df["IDENTITY"].astype(str))

num_samples = len(df)
image_data_memmap = np.memmap("image_data.npy", dtype=np.float32, mode="w+", shape=(num_samples, *image_shape))
label_data_memmap = np.memmap("label_data.npy", dtype=object, mode="w+", shape=(num_samples,))

def load_and_process_image(image_path, target_size=(64, 256)):
    try:
        img = Image.open(image_path).convert("L")  
        img = img.resize((target_size[1], target_size[0]))  
        img_array = np.array(img, dtype=np.float32) / 255.0  
        if img_array.shape != (target_size[0], target_size[1]):
            print(f"Warning: Image {image_path} has shape {img_array.shape}, resizing again.")
            img_array = np.resize(img_array, target_size)

        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None
def process_batch(batch_df, start_idx):
    for i, (_, row) in enumerate(tqdm(batch_df.iterrows(), total=len(batch_df))):
        img_path = os.path.join(image_dir, row["FILENAME"])
        img_array = load_and_process_image(img_path)

        if img_array is not None:
            image_data_memmap[start_idx + i] = img_array
            label_data_memmap[start_idx + i] = tokenizer.texts_to_sequences([row["IDENTITY"]])[0]

for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i : i + batch_size]
    process_batch(batch_df, start_idx=i)

image_data_memmap.flush()
label_data_memmap.flush()

print("Processing complete. Data saved to image_data.npy and label_data.npy.")

