from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow as tf
import pandas as pd
# Define the same character set used during training
CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Load the tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(list(CHARACTERS))  

import tensorflow.keras.backend as K

# def ctc_decode(predictions):
#     input_length = np.ones(predictions.shape[0]) * predictions.shape[1]
#     results, _ = K.ctc_decode(predictions, input_length, greedy=True)
#     print("results......",results[0].shape)
#     decoded_texts = []
#     for result in results[0].numpy():
#         decoded_text = "".join([CHARACTERS[idx - 1] for idx in result if idx > 0])
#         decoded_texts.append(decoded_text)
    
#     return decoded_texts
# import keras.backend as K

def decode_ctc(predictions):
    decoded, _ = K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1])
    return ["".join([CHARACTERS[i] for i in seq if i >= 0]) for seq in decoded[0].numpy()]



load_model = tf.keras.models.load_model('ocr_model.keras', compile=False)
df= pd.read_csv("dataset/written_name_test_v2.csv")
test_images = "dataset/test_v2/test/"+df['FILENAME']
test_labels = df["IDENTITY"]
# Run model prediction
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(208, 52)):
    img = Image.open(image_path).convert("L") 
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0 
    img_array = np.expand_dims(img_array, axis=-1) 
    return img_array

test_images_array = np.array([load_and_preprocess_image(img) for img in test_images]) 
print(test_images_array.shape)
print(test_images_array.dtype) 

predictions = load_model.predict(test_images_array)
print(predictions[0])
decoded_texts = decode_ctc(predictions)  
print("decoded",decoded_texts)
for i in range(5):
    print(f"Predicted: {decoded_texts[i]}, Actual: {test_labels[i]}")


from jiwer import wer, cer
import ast

def safe_eval(label):
    """Safely convert a string label to a list of integers, handling errors."""
    try:
        return ast.literal_eval(label) if isinstance(label, str) and label.startswith("[") else label
    except (SyntaxError, ValueError):
        print(f"Warning: Skipping invalid label -> {label}")  # Debugging message
        return []  # Return an empty list if conversion fails

# Apply safe conversion
test_labels = [safe_eval(label) for label in test_labels]

# Now, decode properly
actual_texts = ["".join([CHARACTERS[idx - 1] for idx in label if idx > 0]) for label in test_labels]
# Compute error rates
print(f"Character Error Rate (CER): {cer(actual_texts, decoded_texts):.4f}")
print(f"Word Error Rate (WER): {wer(actual_texts, decoded_texts):.4f}")
