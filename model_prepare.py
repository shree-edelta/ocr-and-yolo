import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Function to remove extensions and get the base name
def remove_extensions(file_name):
    base_name = os.path.splitext(file_name)[0]
    while '.' in base_name:
        base_name = os.path.splitext(base_name)[0]
    return base_name

folder_path = 'archive 2/Template1/training_data'  

all_files = os.listdir(folder_path)

tiff_files = []
gt_txt_files = []
labels = []
for file in all_files:
    if file.endswith('.tiff'):
        tiff_files.append(file)
    elif file.endswith('.gt.txt'):
        gt_txt_files.append(file)

# Label and image processing
for tiff_file in tiff_files:
    tiff_base = remove_extensions(tiff_file)
    matching_txt_file = None
    
    for txt_file in gt_txt_files:
        if remove_extensions(txt_file) == tiff_base:
            matching_txt_file = txt_file
            break

    if matching_txt_file:
        with open(os.path.join(folder_path, matching_txt_file), 'r') as file:
            label = file.read().strip()
            labels.append(label)

def load_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert('RGB') 
    image = image.resize(target_size)  
    image = np.array(image)  
    image = image / 255.0  
    return image

images = []
for tiff_file in tiff_files:
    base_name = remove_extensions(tiff_file)
    matching_txt_file = base_name + '.gt.txt'
    if matching_txt_file in all_files:
        img_path = os.path.join(folder_path, tiff_file)
        image = load_image(img_path)
        images.append(image)

X = np.array(images)


label_encoder = LabelEncoder()
all_characters = set(''.join(labels))
label_encoder.fit(sorted(all_characters))

y_encoded = [label_encoder.transform(list(label)) for label in labels]
# print(y_encoded)

max_label_length = max(len(label) for label in y_encoded) 
# print(max_label_length)

y_encoded_padded = [label.tolist() + [0] * (max_label_length - len(label)) if len(label) < max_label_length else label.tolist() for label in y_encoded]

# y_encoded = [label + [0] * (max_label_length - len(label)) for label in y_encoded]

print(y_encoded_padded)
y = np.array(y_encoded_padded)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


num_classes = len(label_encoder.classes_)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Reshape((-1, 128)),  
    
    layers.LSTM(128, return_sequences=True, dropout=0.25),
    layers.Dropout(0.25),
    
    layers.Dense(num_classes, activation=None)  # Raw logits, no softmax
])

# CTC loss function
def ctc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    label_lengths = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
    logit_lengths = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])  # Time steps in predictions
    return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_lengths, logit_length=logit_lengths))

# Compile the model
model.compile(optimizer='adam', loss=ctc_loss)
model.summary()
# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')