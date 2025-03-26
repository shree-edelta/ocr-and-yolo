# from PIL import Image
# import pytesseract

# # If Tesseract is not in your PATH, specify its installation location
# # Example for Windows (adjust the path if necessary)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load the image using Pillow
# image = Image.open('dataset/test_v2/test/TEST_0002.jpg')

# # Use pytesseract to do OCR on the image
# extracted_text = pytesseract.image_to_string(image)

# # Print the extracted text
# print("Extracted Text: ", extracted_text)


# import cv2
# import pytesseract

# # Load the image
# image = cv2.imread('dataset/test_v2/test/TEST_0002.jpg')

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Optionally, apply thresholding to binarize the image
# _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# # Use pytesseract to extract text
# extracted_text = pytesseract.image_to_string(thresholded_image)

# print("Extracted Text: ", extracted_text)


# # Save the extracted text to a file
# with open('extracted_text.txt', 'w') as file:
#     file.write(extracted_text)

# # Or print it directly
# print(extracted_text)








# # ?????????????????????????????????????????????????????
# import tensorflow as tf
# from tensorflow.keras import layers, models

# # Define CNN-RNN architecture for handwritten recognition
# def cnn_rnn_model(input_shape=(32, 32, 1), num_classes=26):
#     model = models.Sequential()

#     # CNN part
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))

#     # Flatten CNN output
#     model.add(layers.Flatten())

#     # RNN part (LSTM)
#     model.add(layers.Reshape((-1, 128)))  # Reshape to fit the LSTM input
#     model.add(layers.LSTM(128, return_sequences=True))
#     model.add(layers.LSTM(128))

#     # Dense and output layer
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dense(num_classes, activation='softmax'))  # For classification (26 classes for letters)

#     return model

# # Example model summary
# model = cnn_rnn_model()
# model.summary()

import cv2 as cv
import numpy as np
from cv2 import te
image = cv.imread('dataset/test_v2/test/TEST_0002.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Image", image)
cv.waitKey(0)