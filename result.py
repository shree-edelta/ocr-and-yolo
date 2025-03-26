import os
import glob as glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
import random

seed = 42
np.random.seed(seed)
 
train = True
epochs = 25

def set_res_dir():
    res_dir_count = len(glob.glob('runs/train/*'))
    print(f"current numbrer of results: {res_dir_count}")
    if train:
        res_dir = f"runs/train/{res_dir_count}"
    else:
        res_dir = f"runs/val/{res_dir_count}"
    return res_dir    

# def monitor_tensorboard():
#     %load_ext tensorboard
#     %tensorboard --logdir runs/train     
    
    
# res_dir = set_res_dir()
# if train:
#     !python train.py --data acad_project.v4i.yolov5pytorch/data.yaml --weights yolov5s.pt \
#     --img 640 --batch-size 16 --epochs 30  --name {res_dir}
