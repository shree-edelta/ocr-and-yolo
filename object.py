import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_form_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Apply dilation to connect gaps in the boxes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected boxes
    output_image = image.copy()

    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small boxes (optional, to focus on form fields)
        if w > 50 and h > 50:
            # Draw bounding box around the detected form field
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the result
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test the function
image_path = 'images/hform.jpg'
detect_form_boxes(image_path)
