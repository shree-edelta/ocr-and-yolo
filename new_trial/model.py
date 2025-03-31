# prepare data

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Load processed dataset
train_df = pd.read_csv("csvdata/train_process_data.csv")
val_df = pd.read_csv("csvdata/val_process_data.csv")
test_df = pd.read_csv("csvdata/test_process_data.csv")

# Convert image arrays back to numpy format
train_images = np.array([np.array(eval(img)).reshape((52, 208, 1)) for img in train_df['image_array']])
val_images = np.array([np.array(eval(img)).reshape((52, 208, 1)) for img in val_df['image_array']])
test_images = np.array([np.array(eval(img)).reshape((52, 208, 1)) for img in test_df['image_array']])

# Tokenize labels (character-level encoding)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(train_df['label'])  # Train tokenizer on training labels

train_labels = tokenizer.texts_to_sequences(train_df['label'])
val_labels = tokenizer.texts_to_sequences(val_df['label'])
test_labels = tokenizer.texts_to_sequences(test_df['label'])

# Define max label length (needed for padding)
max_label_len = max(map(len, train_labels))

# Pad sequences to have uniform label length
train_labels = tf.keras.preprocessing.sequence.pad_sequences(train_labels, maxlen=max_label_len, padding="post")
val_labels = tf.keras.preprocessing.sequence.pad_sequences(val_labels, maxlen=max_label_len, padding="post")
test_labels = tf.keras.preprocessing.sequence.pad_sequences(test_labels, maxlen=max_label_len, padding="post")
print(train_labels[0].shape)
# Convert to TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
print("train_dataset element spec",train_dataset.element_spec)

# build ocr

from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

# Define input shape
input_shape = (52, 208, 1)  # Grayscale images

# Input layer
inputs = layers.Input(shape=input_shape, name="image")

# CNN Feature Extractor
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Reshape for LSTM
x = layers.Reshape(target_shape=(-1, 128))(x)

# BiLSTM Layers
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

# Dense Layer for character prediction
x = layers.Dense(len(tokenizer.word_index) + 1, activation="softmax")(x)

# Define CTC loss function
labels = layers.Input(name="label", shape=(max_label_len,))
label_length = layers.Input(name="label_length", shape=(1,))
input_length = layers.Input(name="input_length", shape=(1,))

def ctc_loss_lambda(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

ctc_loss = layers.Lambda(ctc_loss_lambda, output_shape=(1,), name="ctc")([x, labels, input_length, label_length])

# Define Model
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss)

# Compile Model
model.compile(optimizer="adam")
print(model.summary())


# # train model

# # Define input lengths
# train_input_length = np.ones((len(train_labels), 1)) * (train_images.shape[1] // 4)
# train_label_length = np.expand_dims(np.array([len(label) for label in train_labels]), axis=-1)

# val_input_length = np.ones((len(val_labels), 1)) * (val_images.shape[1] // 4)
# val_label_length = np.expand_dims(np.array([len(label) for label in val_labels]), axis=-1)

# # Train the model
# model.fit(
#     x=[train_images, train_labels, train_input_length, train_label_length],
#     y=np.zeros(len(train_images)),  # Dummy target for CTC loss
#     validation_data=([val_images, val_labels, val_input_length, val_label_length], np.zeros(len(val_images))),
#     batch_size=32,
#     epochs=20
# )

# # Evaluate on test data
# # test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# # print(f"Test Loss: {test_loss:.4f}")
# # print(f"Test Accuracy: {test_accuracy:.4f}")


# import numpy as np
# import Levenshtein

# # Evaluate model
# test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# # Calculate Character Error Rate (CER)
# def calculate_cer(predictions, actuals):
#     total_errors, total_chars = 0, 0
#     for pred, act in zip(predictions, actuals):
#         total_errors += Levenshtein.distance(pred, act)
#         total_chars += len(act)
#     return total_errors / total_chars

# # Calculate Word Error Rate (WER)
# def calculate_wer(predictions, actuals):
#     total_errors, total_words = 0, 0
#     for pred, act in zip(predictions, actuals):
#         pred_words, act_words = pred.split(), act.split()
#         total_errors += Levenshtein.distance(" ".join(pred_words), " ".join(act_words))
#         total_words += len(act_words)
#     return total_errors / total_words

# # Calculate Sequence Accuracy
# def sequence_accuracy(predictions, actuals):
#     correct = sum(1 for pred, act in zip(predictions, actuals) if pred == act)
#     return correct / len(actuals)
# import numpy as np

# # Define character mapping (ensure it matches your training tokenizer)
# CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Example
# CHAR_MAP = {i: char for i, char in enumerate(CHARACTERS)}
# BLANK_INDEX = len(CHAR_MAP)  # If using CTC Loss

# def ctc_decode(predictions):
#     """
#     Convert softmax output to text using CTC decoding.
#     """
#     decoded_texts = []
    
#     for pred in predictions:
#         # Convert probability distributions to character indices
#         char_indices = np.argmax(pred, axis=1)  # Get highest probability index per timestep
        
#         # Remove duplicate consecutive characters & blanks
#         decoded_text = []
#         prev_char = None
#         for idx in char_indices:
#             if idx != prev_char and idx != BLANK_INDEX:  # Ignore repeated characters and blank index
#                 decoded_text.append(CHAR_MAP.get(idx, ""))  # Map index to character
#             prev_char = idx

#         decoded_texts.append("".join(decoded_text))
    
#     return decoded_texts

# # Get model predictions
# predictions = model.predict(test_images)
# decoded_predictions = ctc_decode(predictions)  # Ensure you have a decoding function

# # Compute error metrics
# cer = calculate_cer(decoded_predictions, test_labels)
# wer = calculate_wer(decoded_predictions, test_labels)
# seq_acc = sequence_accuracy(decoded_predictions, test_labels)

# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Character Error Rate (CER): {cer:.4f}")
# print(f"Word Error Rate (WER): {wer:.4f}")
# print(f"Sequence Accuracy: {seq_acc:.4f}")

