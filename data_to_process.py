import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

df = pd.read_csv('dataset/written_name_train_v2.csv')
train_images = "dataset/train_v2/train/"+df['FILENAME']
train_labels = df['IDENTITY']


dft = pd.read_csv('dataset/written_name_test_v2.csv')
test_images = "dataset/test_v2/test/"+dft['FILENAME']
test_labels = dft['IDENTITY']

dfv = pd.read_csv('dataset/written_name_validation_v2.csv')
val_images = "dataset/validation_v2/validation/"+dfv['FILENAME']
val_labels = dfv['IDENTITY']


# preprocess
def load_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

def process_csv(image_path, label):
    images_normalized = []
    for i in image_path:
        
        img_array = np.array(load_image(i))
        images_normalized.append(img_array)
  
    
    label = [str(label) for label in label]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(label)
    encoded_labels = tokenizer.texts_to_sequences(label)
   
    df = pd.DataFrame(list(zip(images_normalized, encoded_labels)), columns=['images', 'labels']) 
    return df
        
train= process_csv(train_images, train_labels)
train.to_csv('train_process_data.csv')
val= process_csv(val_images, val_labels)
val.to_csv('val_process_data.csv')


test= process_csv(test_images, test_labels)
test.to_csv('test_process_data.csv')