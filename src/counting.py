import cv2

# Count detected objects
def count_objects(detections):
    """Counts the total number of objects detected."""
    return len(detections)

# Visualize detections
def visualize_detections(image, detections, target_classes, class_names):
    """Draw bounding boxes and labels on the image for detected objects."""
    for (x1, y1, x2, y2, confidence, class_id) in detections:
        label = class_names[class_id]
        if class_id in target_classes:
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label and confidence score
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
