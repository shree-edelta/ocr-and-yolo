# import tensorflow as tf
# import pandas as pd
# from tensorflow.keras import layers, models
# from tensorflow.keras.optimizers import Adam
# from keras.utils import to_categorical
# from tensorflow.keras import backend as K
# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# df_test = pd.read_csv('dataset/written_name_test_v2.csv')
# test_images = "dataset/test_v2/test" + df_test['FILENAME']
# test_labels = df_test['IDENTITY']

# def convert_data(file_name):   
#     data = pd.read_csv(file_name)
   
#     f_array = []
#     label_list = []  
    
#     for i in range(len(data['images'])):
#         array_str = data['images'][i]  
#         str_list = data['labels'][i]
        
#         str_list = str_list.strip().strip('[]').split(',') 
#         int_list = list(map(int, str_list))
        
#         array_str_cleaned = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
#         array_list = np.fromstring(array_str_cleaned, sep=' ')
#         array_list = array_list.reshape((3, 3))  
#         array = np.array(array_list)
#         f_array.append(array)
#         label_list.append(int_list)
    
#     images_array = np.array(f_array)
#     images_array = images_array.astype('float32') / 255.0  
#     images_array = np.expand_dims(images_array, axis=-1) 

#     return images_array, label_list

    
   

# def build_handwriting_recognition_model(img_height, img_width, num_classes):
   
#     input_img = layers.Input(shape=(img_height, img_width, 1)) 

#     # CNN layers
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Reshape the feature maps to be compatible with RNN
#     new_shape = (x.shape[1], x.shape[2] * x.shape[3])  
#     x = layers.Reshape(target_shape=new_shape)(x)

#     # RNN layers (LSTM)
#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
#     # Output layer
#     output = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs=input_img, outputs=output)
#     # print(model.output_shape)
#     return model


# def ctc_loss(y_true, y_pred):
    
#     # print("000000090000000000000",y_pred[0])
#     print("y_pred:",y_pred.shape)
#     print("y_true:",y_true.shape)
#     # y_true = tf.reshape(y_true, -1)      
#     # y_pred = tf.reshape(y_pred, -1)
#     y_true = tf.cast(y_true, dtype=tf.int32)
#     y_pred = tf.cast(y_pred, dtype=tf.float32)
#     batch_size = tf.shape(y_pred)[0]
#     input_length = tf.ones([batch_size], dtype=tf.int32) * tf.shape(y_pred)[1] 
#     label_length = tf.ones([batch_size], dtype=tf.int32) * tf.shape(y_true)[1] 
    
    
#     loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_length, logit_length=input_length))

    

    
#     # Compute the CTC loss
#     print("y_true shape:", tf.shape(y_true))
#     print("y_pred shape:", tf.shape(y_pred))
#     print("input_length shape:", tf.shape(input_length))
#     print("label_length shape:", tf.shape(label_length))
#     print("y_true:", y_true)
#     print("y_pred:", y_pred)
#     print("input_length:", input_length)
#     print("label_length:", label_length)
    
#     # return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_length, logit_length=input_length))
    
#     # loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
                  
#     # label_length = K.sum(K.cast(K.not_equal(y_true_flattened, 0), dtype=tf.int32), axis=-1)
#     # input_length = K.ones_like(label_length) * K.shape(y_pred_flattened)[1]  # time_steps in y_pred
    
#     # batch_size = tf.shape(y_pred)[0]
#     # input_length = tf.ones([batch_size], dtype=tf.float32) * tf.cast(tf.shape(y_pred)[1], tf.float32)  # Time steps (4 in your case)
#     # # label_length = [len(label) for label in train_labels]
#     # # Length of the true sequences (y_true)
#     # label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype=tf.int32), axis=-1)
    
#     # input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1]  # All sequences have the same length
#     # # label_length = np.array([len(label) for label in y_true])
#     # label_length = K.sum(K.cast(K.not_equal(y_true, 0), dtype=tf.int32), axis=-1)
#     # batch_size = tf.shape(y_pred)[0]
#     # input_length = np.ones(batch_size) * 4 
#     # label_length = np.array([len(label) for label in y_true])
    
    
#     return loss
#     # Calculate CTC loss
#     # return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    
   
    
#     # label_length = K.sum(K.cast(K.not_equal(y_true, 0), dtype=tf.int32), axis=-1)
#     # input_length = K.ones_like(label_length) * K.shape(y_pred)[1] 
    
#     # input_length = tf.ones_like(y_true, dtype=tf.int32) * tf.shape(y_pred)[1]
#     # label_length = K.sum(K.cast(K.not_equal(y_true, 0), dtype=tf.int32), axis=-1)
#     # return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)




# def resize_images(images, target_height, target_width):
#     resized_images = []
#     for img in images:
#         img_resized = tf.image.resize(img, (target_height, target_width))
#         img_resized = tf.image.convert_image_dtype(img_resized, tf.float32)
#         img_resized /= 255.0
#         resized_images.append(img_resized)
#     return np.array(resized_images)

# img_height=32 
# img_width = 128  
# num_classes = 31  
# img_channels = 3

# train_images, train_labels = convert_data('dataset/train_process_data.csv')
# val_images, val_labels = convert_data('dataset/val_process_data.csv')


# train_images_resized = resize_images(train_images, 32, 128)
# val_images_resized = resize_images(val_images, 32, 128)

# max_label_length = max(len(label) for label in train_labels)
# # train_labels_padded = pad_sequences(train_labels, padding='post', value=0)
# train_labels_padded = pad_sequences(train_labels, maxlen=max_label_length, padding='post', value=0)

# print("train label padded",train_labels_padded.shape)
# train_labels = np.array(train_labels_padded)
# # label_length = [len(label) for label in train_labels]

# # val_labels_padded = pad_sequences(val_labels, padding='post', value=0)
# val_labels_padded = pad_sequences(val_labels, maxlen=max_label_length, padding='post', value=0)
# val_labels = np.array(val_labels_padded)

# print(f"Train images shape: {train_images_resized.shape}, type: {train_images_resized.dtype}")
# # print(f"Train labels shape: {train_labels.shape}, type: {train_labels.dtype}")
# print(train_labels[0])
# print(train_labels.dtype,train_labels.shape)
# print(f"Validation images shape: {val_images_resized.shape}, type: {val_images_resized.dtype}")
# # print(f"Validation labels shape: {val_labels.shape}, type: {val_labels.dtype}")


# model = build_handwriting_recognition_model(img_height, img_width, num_classes)
# print(model.summary())

# model.compile(optimizer=Adam(), loss=ctc_loss)
# model.fit(train_images_resized, train_labels, epochs=10, batch_size=32, validation_data=(val_images_resized, val_labels))

# # model.summary()
# # model.save('cnn_rnn_model.h5')

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test Accuracy: {test_acc}')


# y_pred = model.predict(test_images)
# print(y_pred.shape)








import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Model architecture
def build_handwriting_recognition_model(img_height, img_width, num_classes):
   
    input_img = layers.Input(shape=(img_height, img_width, 1)) 

    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape the feature maps to be compatible with RNN
    new_shape = (x.shape[1], x.shape[2] * x.shape[3])  
    x = layers.Reshape(target_shape=new_shape)(x)

    # RNN layers (LSTM)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_img, outputs=output)
    return model


# Custom CTC loss function
def ctc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)  
    y_pred = tf.cast(y_pred, dtype=tf.float32) 

    batch_size = tf.shape(y_pred)[0] 
    time_steps = tf.shape(y_pred)[1]  # Time steps (width of input image after CNN + Pooling layers)
    num_classes = tf.shape(y_pred)[2]  # Number of classes (output size of Dense layer)

    # Compute the length of each sequence in the batch
    input_length = tf.ones([batch_size], dtype=tf.int32) * time_steps  # All sequences have the same time steps (width of the image)

    # Compute the length of each label sequence
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype=tf.int32), axis=-1)  # Length of each label sequence (without padding)

    # Compute CTC loss
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=label_length, logit_length=input_length, logits_time_major=False))

    return loss

# Image resizing function
def resize_images(images, target_height, target_width):
    resized_images = []
    for img in images:
        img_resized = tf.image.resize(img, (target_height, target_width))
        img_resized = tf.image.convert_image_dtype(img_resized, tf.float32)
        img_resized /= 255.0
        resized_images.append(img_resized)
    return np.array(resized_images)

# Set model parameters
img_height = 32
img_width = 128  
num_classes = 37  
img_channels = 3

# Loaddata
train_images, train_labels = convert_data('dataset/train_process_data.csv')
val_images, val_labels = convert_data('dataset/val_process_data.csv')

# Resize the images to the required input size for the model
train_images_resized = resize_images(train_images, 32, 128)
val_images_resized = resize_images(val_images, 32, 128)

# Pad the labels to the maximum length
max_label_length = max(len(label) for label in train_labels)
train_labels_padded = pad_sequences(train_labels, maxlen=max_label_length, padding='post', value=0)
train_labels = np.array(train_labels_padded)

val_labels_padded = pad_sequences(val_labels, maxlen=max_label_length, padding='post', value=0)
val_labels = np.array(val_labels_padded)


print(f"Train images shape: {train_images_resized.shape}, type: {train_images_resized.dtype}")
print(f"Train labels shape: {train_labels.shape}, type: {train_labels.dtype}")
print(f"Validation images shape: {val_images_resized.shape}, type: {val_images_resized.dtype}")
print(f"Validation labels shape: {val_labels.shape}, type: {val_labels.dtype}")

# model
model = build_handwriting_recognition_model(img_height, img_width, num_classes)
model.compile(optimizer=Adam(), loss=ctc_loss)

# Train the model
model.fit(train_images_resized, train_labels, epochs=10, batch_size=32, validation_data=(val_images_resized, val_labels))
model.save('cnn_rnn_model.h5')

# Evaluate 
test_images, test_labels = convert_data('dataset/test_process_data.csv')

test_images_resized = resize_images(test_images, 32, 128) 
test_labels_padded = pad_sequences(test_labels, maxlen=max_label_length, padding='post', value=0)  


test_loss = model.evaluate(test_images_resized, test_labels_padded)
print(f'Test Loss: {test_loss}')

# predictions on test data
y_pred = model.predict(test_images_resized)
print(f"Predicted shape: {y_pred.shape}")
