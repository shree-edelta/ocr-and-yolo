import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.backend import ctc_decode

# Reading the test data
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


# def build_handwriting_recognition_model(img_height, img_width, num_classes):
   
#     input_img = layers.Input(shape=(img_height, img_width, 1)) 

#     # CNN layers
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     new_shape = (x.shape[1], x.shape[2] * x.shape[3])  
#     x = layers.Reshape(target_shape=new_shape)(x)

#     # RNN layers (LSTM)
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

#     output = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs=input_img, outputs=output)
#     return model

# def build_handwriting_recognition_model(img_height, img_width, num_classes):
    
#     input_img = layers.Input(shape=(img_height, img_width, 1))  
#     print(input_img.shape)
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     print(f"Shape after CNN layers: {x.shape}")
#     # x = layers.Reshape((-1, x.shape[-1]))(x)
#     # x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1], -1)))(x)
#     x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1] * tf.shape(t)[2], tf.shape(t)[-1])))(x)

#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

#     # output = layers.Dense(num_classes, activation='softmax')(x) 
#     output = layers.Dense(num_classes)(x)
#     print(f"Output shape: {output.shape}")
#     model = models.Model(inputs=input_img, outputs=output)
#     print(model.output_shape)
#     print(model.summary())
#     return model


# # def ctc_loss(y_true, y_pred):
# #     y_true = tf.cast(y_true, dtype=tf.int32)  
# #     y_pred = tf.cast(y_pred, dtype=tf.float32) 

# #     batch_size = tf.shape(y_pred)[0] 
# #     time_steps = tf.shape(y_pred)[1]  
# #     num_classes = tf.shape(y_pred)[2] 
    
# #     input_length = tf.ones([batch_size], dtype=tf.int32) * time_steps  
# #     label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype=tf.int32), axis=-1)  

# #     loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_length, logit_length=input_length, logits_time_major=False))

# #     return loss

# def ctc_loss(y_true, y_pred):
#     batch_size = tf.shape(y_pred)[0] 
#     time_steps = tf.shape(y_pred)[1]  
#     input_length = tf.fill([batch_size], time_steps)  

#     label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype=tf.int32), axis=-1)

#     return tf.reduce_mean(
#         tf.nn.ctc_loss(
#             labels=y_true,
#             logits=y_pred,
#             label_length=label_length,
#             logit_length=input_length,
#             logits_time_major=False
#         )
#     )

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def ctc_lambda_loss(args):
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def build_handwriting_recognition_model(img_height, img_width, num_classes):
    # Input image
    input_img = layers.Input(shape=(img_height, img_width, 1), name="image_input")

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Fix LSTM input shape
    x = layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1] * tf.shape(t)[2], tf.shape(t)[-1])))(x)

    # Recurrent layers (LSTM for sequence processing)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(num_classes, name="logits")(x)  # No softmax

    # Additional Inputs for CTC Loss
    y_true = layers.Input(name="y_true", shape=(None,), dtype="int32")
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int32")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int32")

    # CTC Loss Layer (Lambda)
    loss_out = layers.Lambda(ctc_lambda_loss, name="ctc_loss")([y_true, x, input_length, label_length])

    # Define the full model
    model = models.Model(inputs=[input_img, y_true, input_length, label_length], outputs=loss_out)

    return model
def ctc_loss(y_true, y_pred):
    """Custom CTC loss function for handwriting recognition."""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def resize_images(images, target_height, target_width):
    images_resized = tf.image.resize(images, (target_height, target_width))
    images_resized = images_resized / 255.0
    return images_resized
    # resized_images = []
    # for img in images:
    #     # images_resized = tf.image.resize(images, (target_height, target_width))
    #     # images_resized = images_resized / 255.0
    #     # return images_resized
    #     img_resized = tf.image.resize(img, (target_height, target_width))
    #     img_resized = tf.image.convert_image_dtype(img_resized, tf.float32)
    #     img_resized /= 255.0
    #     resized_images.append(img_resized)
    # return np.array(resized_images)


img_height = 52
img_width = 208  
num_classes = 37  
img_channels = 3
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


original_width = img_width  

pooling_factor = 2 * 2 * 2  # 3 layers → 2x2, 2x2, 2x2

final_width = original_width // pooling_factor  

train_input_lengths = np.full((len(train_images_resized),), final_width, dtype=np.int32)
val_input_lengths = np.full((len(val_images_resized),), final_width, dtype=np.int32)

train_label_lengths = np.array([len(label) for label in train_labels], dtype=np.int32)
val_label_lengths = np.array([len(label) for label in val_labels], dtype=np.int32)
# model

model = build_handwriting_recognition_model(img_height, img_width, num_classes)
model.compile(optimizer='adam')


# model.fit(train_images_resized, train_labels, epochs=10, batch_size=34, validation_data=(val_images_resized, val_labels),callbacks = [early_stopping])

model.fit(
    x=[train_images_resized, train_labels, train_input_lengths, train_label_lengths],  # Multiple inputs
    y=np.zeros(len(train_images_resized)),  # Dummy output, since model returns loss directly
    epochs=10,
    batch_size=34,
    validation_data=(
        [val_images_resized, val_labels, val_input_lengths, val_label_lengths], 
        np.zeros(len(val_images_resized))
    ),
    callbacks=[early_stopping]
)

test_images, test_labels = convert_data('dataset/test_process_data.csv')
test_images = test_images[:(len(test_images)//2)]
test_labels = test_labels[:(len(test_labels)//2)]

max_label_length = max(len(label) for label in test_labels)
test_images_resized = resize_images(test_images, 52, 208) 
test_labels_padded = pad_sequences(test_labels, maxlen=max_label_length, padding='post', value=0)  
test_input_lengths = np.full((len(test_images_resized),), final_width, dtype=np.int32)
test_label_lengths = np.array([len(label) for label in test_labels], dtype=np.int32)

# test_loss = model.evaluate(test_images_resized, test_labels_padded)
test_loss = model.evaluate(
    x=[test_images_resized, test_labels, test_input_lengths, test_label_lengths],
    y=np.zeros(len(test_images_resized)),  # Dummy output
    batch_size=32
)

print(f"Test CTC Loss: {test_loss}")
# print(f'Test Loss: {test_loss}')

# predictions on test data
raw_predictions = model.predict(test_images_resized)  # Shape: (batch, time_steps, num_classes)

print(f"Predicted shape: {raw_predictions.shape}")
decoded_predictions, _ = tf.nn.ctc_greedy_decoder(
    inputs=tf.math.log(raw_predictions),  
    sequence_length=[raw_predictions.shape[1]] * len(test_images_resized)  
)

# Convert tensor to numpy array
predicted_sequences = tf.sparse.to_dense(decoded_predictions[0]).numpy()
print(f"Predicted sequences shape: {predicted_sequences.shape}")
# Define your character mapping (modify based on your dataset)
char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Modify this

def decode_to_text(sequence, char_list):
    return ''.join([char_list[idx] for idx in sequence if idx >= 0])  # Ignore blank tokens

# Convert predictions to text
predicted_texts = [decode_to_text(seq, char_list) for seq in predicted_sequences]

# Convert true labels to text
true_texts = [decode_to_text(seq, char_list) for seq in test_labels]  # test_labels should be numeric
print(f"Predicted texts: {predicted_texts}")
print(f"True texts: {true_texts}")

correct_predictions = sum([1 for pred, true in zip(predicted_texts, true_texts) if pred == true])
word_accuracy = correct_predictions / len(true_texts)

print(f"Word Accuracy: {word_accuracy * 100:.2f}%")


model.save('ocr_model2.keras')
print(model.summary())
