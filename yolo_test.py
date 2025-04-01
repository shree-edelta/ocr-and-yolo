from pathlib import Path
import torch
import cv2
import numpy as np
# from models.experimental import attempt_load

# Path to the saved model weights
model_path = Path('runs/train/exp5/weights/best.pt')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/best.pt', force_reload=True)  
model.eval()  # Set model to evaluation mode



image_path = 'images/ff.jpg'
image = cv2.imread(image_path)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
org_h,org_w = rgb_image.shape[:2]
# rig_height, orig_width = image.shape[:2]
resized_image = cv2.resize(rgb_image, (640, 640)) 
print(resized_image.shape)
results = model(resized_image)

predictions = results.xywh[0]
print(predictions)
# conf_threshold = 0.4
# predictions = predictions[predictions[:, 4] > conf_threshold]


# for pred in predictions:
#     x_center, y_center, width, height, conf, cls = pred.tolist() 
#     print(x_center, y_center, width, height, conf, cls)
#     x1, y1 = int((x_center - width / 2) * image.shape[1]), int((y_center - height / 2) * image.shape[0])
#     x2, y2 = int((x_center + width / 2) * image.shape[1]), int((y_center + height / 2) * image.shape[0])
#     print(x1,y1,x2,y2)
#     # Draw rectangle and label
#     label = f'{model.names[int(cls)]} {conf:.2f}'
#     # print(model.names)
#     print("label",label)
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Save or display the result
# cv2.imwrite('output_image.jpg', image)  # Save the image
# cv2.imshow('Detection Result', image)  # Show the image with bounding boxes
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(predictions.shape)
conf_threshold = 0.2
print("split prediction",predictions[:, 4])
filtered_predictions = predictions[predictions[:, 4] > conf_threshold]
print("filter prediction",filtered_predictions)
# Draw bounding boxes and labels on the image
for pred in filtered_predictions:
    # xywh = (xywh-np.mean(xywh))/np.std(xywh)
    # x1, y1, x2, y2 = xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
    print("cls//////",pred)
    x_center, y_center, width, height, conf, cls = pred.tolist()
    x1 = int((x_center - width / 2))  
    y1 = int((y_center - height / 2))  
    x2 = int((x_center + width / 2))  
    y2 = int((y_center + height / 2)) 

    scale_x = org_w / 640
    scale_y = org_h / 640
    
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    label = f'{model.names[int(cls)]} {conf:.2f}'
    cv2.putText(image, label, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("label",label)
# Save or display the result
cv2.imwrite('output_image.jpg', image)  # Save result
cv2.imshow('Detection Result', image)  # Display result
cv2.waitKey(0)
cv2.destroyAllWindows()


# from flask import Flask, request, jsonify
# import torch
# import cv2
# from PIL import Image
# from io import BytesIO

# app = Flask(__name__)

# # Load model
# model = attempt_load('runs/train/exp/weights/best.pt', map_location=torch.device('cpu'))
# model.eval()

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     # Process the image
#     img = Image.open(BytesIO(file.read()))
#     # Preprocessing (resize, convert to tensor, etc.)
#     # Make prediction...
#     # Return the result (image or JSON)

#     return jsonify({'message': 'Prediction successful!'})

# if __name__ == '__main__':
    # app.run(debug=True)
