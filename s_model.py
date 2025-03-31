import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from jiwer import cer, wer

# --------------------- MODEL DEFINITION ---------------------
class HandwritingRecognitionModel(keras.Model):
    def __init__(self, img_height, img_width, num_classes):
        super().__init__()
        self.conv1 = layers.Conv2D(32, (3,3), activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = layers.Conv2D(64, (3,3), activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.reshape = layers.Reshape((-1, 64))
        self.bi_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dense = layers.Dense(num_classes + 1, activation="softmax")  # Extra class for CTC blank token
    
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
            y_pred = self(images, training=True)
            loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": tf.reduce_mean(loss)}

# --------------------- DATA PROCESSING ---------------------
def resize_images(images, target_height, target_width):
    images_resized = tf.image.resize(images, (target_height, target_width))
    return images_resized / 255.0

# def convert_data(file_name):
#     data = pd.read_csv(file_name)
#     images_array = np.array([np.fromstring(img.replace('[', '').replace(']', ''), sep=' ').reshape((52, 208, 1))
#                              for img in data['images']])
#     labels = [list(map(int, label.strip('[]').split(','))) for label in data['labels']]
#     return images_array.astype('float32') / 255.0, labels
def convert_data(file_name):
    data = pd.read_csv(file_name)
    f_array, label_list = [], []  
    
    for i in range(len(data['images'])):
        array_str = data['images'][i]
        array_list = np.fromstring(array_str.replace('[', '').replace(']', ''), sep=' ')
        
        if len(array_list) != 52 * 208:  # Check size before reshaping
            print(f"Skipping incorrect image at index {i}, size {len(array_list)}")
            continue
        
        array_list = array_list.reshape((52, 208, 1))
        f_array.append(array_list)
        label_list.append([int(x) for x in data['labels'][i].strip().strip('[]').split(',')])
    
    return np.array(f_array), np.array(label_list)


# --------------------- TRAINING SETUP ---------------------
img_height, img_width, num_classes = 52, 208, 37
train_images, train_labels = convert_data('csvdata/train_process_data.csv')
val_images, val_labels = convert_data('csvdata/val_process_data.csv')

# train_images = train_images[:len(train_images)//2]
# train_labels = train_labels[:len(train_labels)//2]
# val_images = val_images[:len(val_images)//2]
# val_labels = val_labels[:len(val_labels)//2]

max_label_length = max(len(label) for label in train_labels)
train_labels_padded = pad_sequences(train_labels, maxlen=max_label_length, padding='post', value=0)
val_labels_padded = pad_sequences(val_labels, maxlen=max_label_length, padding='post', value=0)
train_input_lengths = np.ones((len(train_images),)) * (img_width // 8)
val_input_lengths = np.ones((len(val_images),)) * (img_width // 8)
train_label_lengths = np.array([len(label) for label in train_labels], dtype=np.int32)
val_label_lengths = np.array([len(label) for label in val_labels], dtype=np.int32)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --------------------- TRAINING ---------------------
model = HandwritingRecognitionModelWithCTC(img_height, img_width, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model.fit(
    x=(train_images, train_labels_padded, train_input_lengths, train_label_lengths),
    batch_size=32,
    epochs=10,
    validation_data=(val_images, val_labels_padded, val_input_lengths, val_label_lengths),
    callbacks=[early_stopping]
)

# --------------------- EVALUATION ---------------------
test_images, test_labels = convert_data('csvdata/test_process_data.csv')
test_images = test_images[:len(test_images)//2]
test_labels = test_labels[:len(test_labels)//2]
test_labels_padded = pad_sequences(test_labels, maxlen=max_label_length, padding='post', value=0)
test_input_lengths = np.ones((len(test_images),)) * (img_width // 8)
test_label_lengths = np.array([len(label) for label in test_labels], dtype=np.int32)

model.evaluate((test_images, test_labels_padded, test_input_lengths, test_label_lengths))

def decode_predictions(predictions):
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    results = K.ctc_decode(predictions, input_length=input_len, greedy=True)[0][0]
    return results.numpy()

def numerical_to_text(numerical_sequences, char_map):
    return ["".join([char_map[i] for i in sequence if i in char_map]) for sequence in numerical_sequences]

characters = "abcdefghijklmnopqrstuvwxyz0123456789"
char_map = {i: char for i, char in enumerate(characters)}

raw_predictions = model.predict(test_images)
decoded_predictions = decode_predictions(raw_predictions)
cer_score = cer(numerical_to_text(decoded_predictions, char_map), numerical_to_text(test_labels, char_map))
wer_score = wer(numerical_to_text(decoded_predictions, char_map), numerical_to_text(test_labels, char_map))

print(f"Character Error Rate (CER): {cer_score:.2%}")
print(f"Word Error Rate (WER): {wer_score:.2%}")

# --------------------- SAVE MODEL ---------------------
if cer_score < 0.3:  # Save model only if accuracy is reasonable
    model.save('ocr_model2.keras')
    print("Model saved successfully!")

print(model.summary())
