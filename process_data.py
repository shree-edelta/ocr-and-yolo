import pandas as pd
import numpy as np
import ast
def convert_data(file_name):   
    data = pd.read_csv(file_name)
    data = data.dropna()
   
    f_array=[]
    for i in range(len(data['images'])):
        array_str = data['images'][i]  
        str_list = data['labels'][i]
        
        str_list = str_list.strip().strip('[]').split(',') 
        int_list = list(map(int, str_list))
        
        array_str_cleaned = array_str.replace('[', '').replace(']', '').replace('\n', ' ')
        array_list = np.fromstring(array_str_cleaned, sep=' ')
        array_list = array_list.reshape((3,3))
        array = np.array(array_list)
        f_array.append(array)
    return f_array,int_list


train_image,train_label = convert_data('train_process_data.csv')


val_image,val_label = convert_data('val_process_data.csv')

