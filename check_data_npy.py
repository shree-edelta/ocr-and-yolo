import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the processed dataset
data = np.load("image_data.npy", allow_pickle=True)
images = data["images"], data["labels"]

data2 = np.load("label_data.npy", allow_pickle=True)
labels =  data["labels"]

# Ensure images have the correct shape (64, 256, 1)
images = np.expand_dims(images, axis=-1)  # Add channel dimension

# Padding labels to a fixed length
max_label_length = max(len(label) for label in labels)
labels = pad_sequences(labels, maxlen=max_label_length, padding="post", value=0)

print(f"Loaded {len(images)} images and {len(labels)} labels.")
