import cv2
from src.yolo_detect import load_yolo_model, detect_objects
from src.counting import count_objects

# COCO class names
CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def main(image_path):
    # Load YOLOv5 model
    model = load_yolo_model()
    print("YOLO model loaded successfully!")

    # Detect objects
    detections, annotated_image = detect_objects(image_path, model, confidence_threshold=0.2)
    if detections is None or annotated_image is None:
        print("Object detection failed. Exiting.")
        return

    # Count objects
    total_objects = count_objects(detections)
    print(f"Total objects detected: {total_objects}")

    # Save the annotated image
    output_path = "outputs/annotated_image.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    # Path to input image
    image_path = "data/poltski.jpg"
    main(image_path)
