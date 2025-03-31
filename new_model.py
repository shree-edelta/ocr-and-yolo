import tensorflow as tf
from tensorflow.keras import Model, layers, backend as K
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class HandwritingRecognitionModel(keras.Model):
    def __init__(self, img_height, img_width, num_classes):
        super().__init__()
        
        # CNN Backbone
        self.conv1 = layers.Conv2D(32, (3,3), activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = layers.Conv2D(64, (3,3), activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        
        # Feature Extraction
        self.reshape = layers.Reshape((-1, 64))
        self.bi_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        
        # Output Layer
        self.dense = layers.Dense(num_classes, activation="softmax")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.reshape(x)
        x = self.bi_lstm(x)
        return self.dense(x)

class HandwritingRecognitionModelWithCTC(HandwritingRecognitionModel):
    def train_step(self, data):
        images, labels, input_length, label_length = data  # Unpack dataset

        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)  # Forward pass
            loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)  # CTC loss

        # Compute Gradients & Apply Updates
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": tf.reduce_mean(loss)}  # Return loss

    def test_step(self, data):
        images, labels, input_length, label_length = data
        y_pred = self(images, training=False)
        loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
        return {"loss": tf.reduce_mean(loss)}

# Create Model


def resize_images(images, target_height, target_width):
    images_resized = tf.image.resize(images, (target_height, target_width))
    images_resized = images_resized / 255.0
    return images_resized
df_test = pd.read_csv('dataset/written_name_test_v2.csv')
test_images = "dataset/test_v2/test" + df_test['FILENAME']
test_labels = df_test['IDENTITY']

def convert_data(file_name):   
    data = pd.read_csv(file_name)
   
    f_array = []
    label_list = []  
    
    for i in range(len(data['images'])):
        array_str = data['images'][i]  
        str_list = data['labels'][i]
        
        str_list = str_list.strip().strip('[]').split(',') 
        int_list = list(map(int, str_list))
        
        array_str_cleaned = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
        array_list = np.fromstring(array_str_cleaned, sep=' ')
        array_list = array_list.reshape((3, 3))  
        array = np.array(array_list)
        f_array.append(array)
        label_list.append(int_list)
    
    images_array = np.array(f_array)
    images_array = images_array.astype('float32') / 255.0  
    images_array = np.expand_dims(images_array, axis=-1) 

    return images_array, label_list
img_height = 52
img_width = 208
num_classes = 37  # A-Z + 0-9

early_stopping = EarlyStopping(monitor='val_loss',  # or 'val_accuracy'
                               patience=5,           # Number of epochs to wait for improvement
                               restore_best_weights=True,  # Restore model with the best validation performance
                               verbose=1)

train_images, train_labels = convert_data('dataset/train_process_data.csv')
val_images, val_labels = convert_data('dataset/val_process_data.csv')

train_images = train_images[:(len(train_images)//2)]
train_labels = train_labels[:(len(train_labels)//2)]
val_images = val_images[:(len(val_images)//2)]
val_labels = val_labels[:(len(val_labels)//2)]
train_images = np.array(train_images)  # Ensure it's an array
train_labels = np.array(train_labels)
print(train_images[0])

train_images_resized = resize_images(train_images, 52, 208)
# print(train_images_resized.shape,train_images_resized.dtype)
val_images_resized = resize_images(val_images, 52, 208)

max_label_length = max(len(label) for label in train_labels)
train_labels_padded = pad_sequences(train_labels, maxlen=max_label_length, padding='post', value=0)
train_labels = np.array(train_labels_padded)

val_labels_padded = pad_sequences(val_labels, maxlen=max_label_length, padding='post', value=0)
val_labels = np.array(val_labels_padded)


print(f"Train images shape: {train_images_resized.shape}, type: {train_images_resized.dtype}")
print(f"Train labels shape: {train_labels.shape}, type: {train_labels.dtype}")
print(f"Validation images shape: {val_images_resized.shape}, type: {val_images_resized.dtype}")
print(f"Validation labels shape: {val_labels.shape}, type: {val_labels.dtype}")


train_input_lengths = np.ones((len(train_images_resized),)) * (img_width // 8) 
val_input_lengths = np.ones((len(val_images_resized),)) * (img_width // 8)  # Adjust based on CNN
train_label_lengths = np.array([len(label) for label in train_labels], dtype=np.int32)
val_label_lengths = np.array([len(label) for label in val_labels], dtype=np.int32)

# model = HandwritingRecognitionModelWithCTC(img_height, img_width, num_classes)
# train_dataset = tf.data.Dataset.from_tensor_slices(
#     (train_images_resized, train_labels, train_input_lengths, train_label_lengths)
# ).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices(
#     (val_images_resized, val_labels, val_input_lengths, val_label_lengths)
# ).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
# Compile model (no need to pass loss separately)
model = HandwritingRecognitionModelWithCTC(img_height=52, img_width=208, num_classes=80)

# Compile Model (NO LOSS FUNCTION NEEDED)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.fit(
    x=(train_images, train_labels, train_input_lengths, train_label_lengths),
    batch_size=32,
    epochs=10,
    validation_data=(val_images, val_labels, val_input_lengths, val_label_lengths)
)

test_images, test_labels = convert_data('dataset/test_process_data.csv')
test_images = test_images[:(len(test_images)//2)]
test_labels = test_labels[:(len(test_labels)//2)]

max_label_length = max(len(label) for label in test_labels)
test_images_resized = resize_images(test_images, 52, 208) 
test_labels_padded = pad_sequences(test_labels, maxlen=max_label_length, padding='post', value=0)  
# test_input_lengths = np.full((len(test_images_resized),), final_width, dtype=np.int32)
# test_label_lengths = np.array([len(label) for label in test_labels], dtype=np.int32)
test_input_lengths = np.ones((len(test_images_resized),)) * (img_width // 8)
test_label_lengths = np.array([len(label) for label in test_labels], dtype=np.int32)
# test_dataset = tf.data.Dataset.from_tensor_slices(
#     (test_images_resized, test_labels_padded, test_input_lengths, test_label_lengths)
#     ).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


# test_loss = model.evaluate(test_images_resized, test_labels_padded)
model.evaluate((test_images, test_labels, test_input_lengths, test_label_lengths))

# Prediction
y_pred = model.predict(test_images)
# print(f"Test Loss: {test_loss}")

# print(f'Test Loss: {test_loss}')
def decode_predictions(predictions):
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]  # Length of each sequence
    results = tf.keras.backend.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0]
    return results.numpy()

# Function to convert numerical predictions to text
def numerical_to_text(numerical_sequences, char_map):
    text_sequences = []
    for sequence in numerical_sequences:
        text_sequences.append("".join([char_map[i] for i in sequence if i in char_map]))  # Convert indexes to chars
    return text_sequences

from jiwer import cer

def calculate_cer(predictions, ground_truths, char_map):
    pred_texts = numerical_to_text(predictions, char_map)
    gt_texts = numerical_to_text(ground_truths, char_map)
    
    return cer(gt_texts, pred_texts)  # Lower is better
from jiwer import wer

def calculate_wer(predictions, ground_truths, char_map):
    pred_texts = numerical_to_text(predictions, char_map)
    gt_texts = numerical_to_text(ground_truths, char_map)
    
    return wer(gt_texts, pred_texts)  # Lower is better
characters = "abcdefghijklmnopqrstuvwxyz0123456789"  # Add special chars if needed

# Create a mapping of characters to numerical indices
char_map = {i: char for i, char in enumerate(characters)}
# Generate predictions
raw_predictions = model.predict(test_images)  # Shape (batch, time_steps, num_classes)

# Decode predictions
decoded_predictions = decode_predictions(raw_predictions)

# Compute CER & WER
cer_score = calculate_cer(decoded_predictions, test_labels, char_map)
wer_score = calculate_wer(decoded_predictions, test_labels, char_map)

print(f"Character Error Rate (CER): {cer_score:.2%}")
print(f"Word Error Rate (WER): {wer_score:.2%}")

# raw_predictions = model.predict(test_images_resized)  # Shape: (batch, time_steps, num_classes)

# print(f"Predicted shape: {raw_predictions.shape}")
# decoded_predictions, _ = tf.nn.ctc_greedy_decoder(
#     inputs=tf.math.log(raw_predictions),  
#     sequence_length=[raw_predictions.shape[1]] * len(test_images_resized)  
# )

# # Convert tensor to numpy array
# predicted_sequences = tf.sparse.to_dense(decoded_predictions[0]).numpy()
# print(f"Predicted sequences shape: {predicted_sequences.shape}")
# # Define your character mapping (modify based on your dataset)
# char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Modify this

# def decode_to_text(sequence, char_list):
#     return ''.join([char_list[idx] for idx in sequence if idx >= 0])  # Ignore blank tokens

# # Convert predictions to text
# predicted_texts = [decode_to_text(seq, char_list) for seq in predicted_sequences]

# # Convert true labels to text
# true_texts = [decode_to_text(seq, char_list) for seq in test_labels]  # test_labels should be numeric
# print(f"Predicted texts: {predicted_texts}")
# print(f"True texts: {true_texts}")

# correct_predictions = sum([1 for pred, true in zip(predicted_texts, true_texts) if pred == true])
# word_accuracy = correct_predictions / len(true_texts)

# print(f"Word Accuracy: {word_accuracy * 100:.2f}%")


model.save('ocr_model2.keras')
print(model.summary())

