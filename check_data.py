import pandas as pd
 
train = pd.read_csv("dataset/train_process_data.csv")
# train = pd.read_csv("new_trial/newproper/train_process_data.csv")
train_i = train["images"]
train_l = train["labels"]

print("Sample data from train_i:", train_i[:3])
print("Type of first element:", type(train_i[0]))

# import ast
# import numpy as np

# def convert_to_array(image_str):
#     try:
#         image_str = image_str.strip().replace("\n", "")
#         array_list = ast.literal_eval(image_str)
#         return np.array(array_list, dtype=np.float32) 
#     except Exception as e:
#         print(f"Error converting image: {e} \nData: {image_str[:100]}...") 
#         return None
         

# train_i = [convert_to_array(img) for img in train_i]
# train_i = [img for img in train_i if img is not None]
# if len(train_i) > 0:
#     print("Shape of first image:", train_i[0].shape)