import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt', force_reload=True)  
model.eval()
# model = torch.load('yolov5/runs/train/exp5/weights/best.pt', weights_only=False)
# model.eval()  

image_path = 'images/ff4.jpg'
image = cv2.imread(image_path)

orig_height, orig_width = image.shape[:2]

resized_image = cv2.resize(image, (640, 640))  
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
results = model(rgb_image)  

predictions = results.xywh[0]  


conf_threshold = 0.4
predictions = predictions[predictions[:, 4] > conf_threshold]
# colors = np.random.uniform(0, 255, size=(len(class_name), 3))
for pred in predictions:
    print("cls//////",pred)
    x_center, y_center, width, height, conf, cls = pred.tolist()

    x1 = int((x_center - width / 2))  
    y1 = int((y_center - height / 2))  
    x2 = int((x_center + width / 2))  
    y2 = int((y_center + height / 2)) 

    scale_x = orig_width / 640
    scale_y = orig_height / 640
    
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  

   
    label = f'{model.names[int(cls)]} {conf:.2f}' 
    print(model.names)
    print(label)# Get class name and confidence
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the result
cv2.imwrite('output_image.jpg', image)  # Save the image with bounding boxes
cv2.imshow('Detection Result', image)  # Show the image with bounding boxes
cv2.waitKey(0)
cv2.destroyAllWindows()


# class_name = ['button', 'checkbx', 'dropdown', 'heading', 'hr', 'image', 'label', 'radiobtn', 'text', 'textbox']

# colors = np.random.uniform(0, 255, size=(len(class_name), 3))

# def yolo2bbox(bboxes):
#     xmin,ymin = bboxes[0]-bboxes[2]/2,bboxes[1]-bboxes[3]/2
#     xmax,ymax = bboxes[0]+bboxes[2]/2,bboxes[1]+bboxes[3]/2
#     return [xmin,ymin,xmax,ymax]


# def plot_box(image, bboxes, labels):
#     class_names = ['button', 'checkbx', 'dropdown', 'heading', 'hr', 'image', 'label', 'radiobtn', 'text', 'textbox']
#     colors = np.random.uniform(0, 255, size=(len(class_names), 3))
#     h, w, _ = image.shape

#     for box_num, box in enumerate(bboxes):
        
#         label = int(labels[box_num]) 
#         if label < 0 or label >= len(class_names):
#             print(f"Invalid label {label} for box {box_num}. Skipping this box.")
#             continue  

#         x1, y1, x2, y2 = yolo2bbox(box)  
#         xmin, ymin, xmax, ymax = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
#         class_label = class_names[label]  
        
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=colors[label], thickness=2)
        
#         font_scale = min(1, max(3, int(w / 500)))
#         font_thickness = min(2, max(10, int(w / 50)))
#         p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
#         tw, th = cv2.getTextSize(class_label, 0, fontScale=font_scale, thickness=font_thickness)[0]
#         p2 = p1[0] + tw, p1[1] - th - 10
        
#         cv2.rectangle(image, p1, p2, colors[label], -1)
#         cv2.putText(image, class_label, (xmin + 1, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

#     return image  


# def plot(image_paths,label_paths,num_samples):
#     all_training_images = glob.glob(image_paths)
#     all_training_labels = glob.glob(label_paths)    #2 0.1697625 0.2905117270788913 0.31552499999999994 0.29850746268656714
#                                                     # 0 0.5997812499999999 0.31716417910447764 0.42088125000000004 0.3304904051172708
#                                                     # 1 0.5888625 0.7489339019189766 0.14123125000000003 0.19722814498933902
#                                                     # 1 0.83804375 0.7609381663113006 0.2958375000000001 0.19456289978678037
#                                                     # 1 0.1326625 0.7715991471215352 0.18976874999999999 0.18390191897654584
#                                                     # 1 0.3845875 0.7849253731343283 0.1858 0.20522388059701493
#     all_training_images.sort()
#     all_training_labels.sort()
    
#     num_images = len(all_training_images)
    
#     plt.figure(figsize=(15,12))
#     for i in range(num_samples):
#         print("num_images",num_images)
#         j = random.randint(0,num_images-1)
#         image = cv2.imread(all_training_images[j])
#         with open(all_training_labels[j],'r') as f:
#             bboxes = []
#             labels = []
#             label_lines = f.readlines()
#             print(label_lines)
#             for label_line in label_lines:
#                 label = label_line[0]
#                 print("label/////",label)
#                 bbox_string = label_line[2:]
#                 print("bbox_string///",bbox_string)
#                 x_c,y_c,w,h = bbox_string.split(' ')
#                 x_c,y_c,w,h = float(x_c),float(y_c),float(w),float(h)
#                 bboxes.append([x_c,y_c,w,h])
#                 labels.append(label)
#                 print("label list//////",labels)
#                 print("bbbox list////",bboxes)
#         image = plot_box(image,bboxes,labels)
#         plt.subplot(2,2,i+1)
#         plt.imshow(image)
#         plt.title('Image '+str(i+1))
#         plt.axis('off')
#         plt.show()
# plot(image_paths="yolov5/acad_project.v4i.yolov5pytorch/images/train/enhanced_3cf4d5e6-cede-4272-aec0-f29790add91a_png.rf.f436b98452bbcbd45e2c6106e10867fb.jpg",label_paths="yolov5/acad_project.v4i.yolov5pytorch/labels/train/enhanced_3cf4d5e6-cede-4272-aec0-f29790add91a_png.rf.f436b98452bbcbd45e2c6106e10867fb.txt",num_samples=1)
                