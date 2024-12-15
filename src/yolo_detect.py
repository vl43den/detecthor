import torch
import cv2
import numpy as np

# Load the YOLOv5 model
def load_yolo_model():
    """Loads the YOLOv5 model using PyTorch."""
    model = torch.hub.load("yolov5", "custom", path="yolov5/yolov5s.pt", source="local")
    return model

# Perform object detection
def detect_objects(image_path, model, confidence_threshold=0.3):
    """
    Detect objects in an image using YOLOv5.
    Returns detected objects and the annotated image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from path {image_path}")
        return None, None

    # Perform inference
    results = model(image)

    # Extract bounding boxes, confidences, and class IDs
    detections = []
    for *box, conf, cls in results.xyxy[0]:  # Box format: [x1, y1, x2, y2, conf, cls]
        if conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            detections.append((x1, y1, x2, y2, float(conf), int(cls)))

    # Annotate the image
    annotated_image = np.array(results.render()[0])

    return detections, annotated_image
