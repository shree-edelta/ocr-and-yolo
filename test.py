import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import cv2 as cv
# import changes as ch
import matplotlib.pyplot as plt

def resize_images(img, target_height, target_width):
    resized_images = []
    img_resized = tf.image.resize(img, (target_height, target_width))
    img_resized = tf.image.convert_image_dtype(img_resized, tf.float32)
    img_resized /= 255.0
    resized_images.append(img_resized)
    return np.array(resized_images)

def load_image(image_array, target_size=(52, 208)):
    img = image_array.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    return img_array


loaded_model = tf.keras.models.load_model('ocr_model.keras', compile=False)


# test_images, test_labels = ch.convert_data('dataset/test_process_data.csv')

# max_label_length = max(len(label) for label in test_labels)
# test_images_resized = ch.resize_images(test_images, 32, 128) 
# test_labels_padded = ch.pad_sequences(test_labels, maxlen=max_label_length, padding='post', value=0)  


# test_loss = loaded_model.evaluate(test_images_resized, test_labels_padded)
# print(f'Test Loss: {test_loss}')

# # predictions on test data
# y_pred = loaded_model.predict(test_images_resized)
# print(f"Predicted shape: {y_pred.shape}")

# img  =  "hform.jpg"
img = cv.imread('images/name.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_r = cv.resize(img_gray, (208, 52))


cv.imshow("img", img)
cv.waitKey(0)

cv.imshow("img_gray", img_r)
cv.waitKey(0)

cv.destroyAllWindows()

print(img.shape)
print(img_r.shape)
# img_resize = resize_images(img, 32, 128)
# print(img_l)
img_r = np.expand_dims(img_r, axis=-1)  

img_r = np.expand_dims(img_r, axis=0) 


predictions = loaded_model.predict(img_r)
print(predictions)

predicted_indices = np.argmax(predictions, axis=-1)  # Get indices of max probability

# Assuming predicted_indices has shape (1, sequence_length), you can reshape or flatten if needed
predicted_indices = predicted_indices.flatten()
print(predicted_indices)
# predicted_class = np.argmax(predictions, axis=-1)
# print(f"Predicted class: {predicted_class}")


import tensorflow as tf
import numpy as np
import cv2 as cv
class_to_char = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
    19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: '0', 27: '1',
    28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: ' ',  
}

def remove_consecutive_duplicates(predictions):
    new_predictions = []
    previous = None
    for prediction in predictions:
        if prediction != previous:
            new_predictions.append(prediction)
        previous = prediction
    return new_predictions

# loaded_model = tf.keras.models.load_model('ocr_model.keras', compile=False)

img = cv.imread('images/name.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_r = cv.resize(img_gray, (208, 52)) 

img_r = np.expand_dims(img_r, axis=-1)  
img_r = np.expand_dims(img_r, axis=0) 
print(img_r.shape)
predictions = loaded_model.predict(img_r)
print(predictions)

print(predictions.shape)
# for i, pred in enumerate(predictions[0]):
    # print(f"Time Step {i}: {pred}")
    
# predicted_indices = np.argmax(predictions[0], axis=-1) 
# print(predicted_indices)
# predicted_indices = predicted_indices.flatten()


# # filtered_predictions = remove_consecutive_duplicates(predicted_indices)

# predicted_string = ''.join([class_to_char.get(index, '') for index in predicted_indices])

# print(f"Predicted String: {predicted_string}")
from tensorflow.keras.backend import ctc_decode
import numpy as np


class_to_char = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 
    19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ' ', 27: 'A',
    28: 'B', 29: 'C', 30: 'D', 31: 'E', 32: 'F', 33: 'G', 34: 'H', 35: 'I', 36: 'J'
   
}


decoded, _ = ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1], greedy=True)
print("decoded.............",len(decoded))
decoded_sequence = decoded[0].numpy() 
print("decoded shape",decoded_sequence.shape)
print(decoded_sequence) 
predicted_text = ''.join([class_to_char.get(int(idx), '') for idx in decoded_sequence.flatten() if idx != -1])

print(predicted_text)


