import cv2
import numpy as np

# Load the MobileNet-SSD model
def load_model():
    """Loads the MobileNet-SSD model."""
    model_path = "models/mobilenet_iter_73000.caffemodel"
    prototxt_path = "models/deploy.prototxt"

    # Load the model using OpenCV
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

# Detect objects in an image
def detect_objects(image_path, net, target_classes, confidence_threshold=0.2):
    """Detects objects in the image."""
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]

    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image from path: {image_path}")
            return None, None

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()

        # Store detected objects
        detected_objects = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                if class_id in target_classes:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    detected_objects.append((x1, y1, x2, y2, confidence, class_id))

        print(f"Detected {len(detected_objects)} objects in the image.")
        return detected_objects, image

    except Exception as e:
        print(f"Error during detection: {e}")
        return None, None
