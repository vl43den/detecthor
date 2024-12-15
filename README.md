# Detecthor: Object Detection with YOLOv5

Detecthor is an object detection project leveraging the power of YOLOv5. It can detect objects in images, count them, and annotate images with bounding boxes and labels. This project is designed for real-world applications and is suitable for both beginners and professionals interested in computer vision.

---

## Features
- **Object Detection**: Uses YOLOv5 to detect objects in images.
- **Object Counting**: Counts the total number of detected objects.
- **Image Annotation**: Saves annotated images with bounding boxes and labels.
- **Customizability**: Modify confidence thresholds and target classes.

---

## Installation

### 1. Clone the Repository

git clone https://github.com/vl43den/detecthor.git
cd detecthor

### 2. Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies

Install all required Python libraries:

pip install -r requirements.txt

Usage

Place Your Input Image Place your image in the data/ directory. By default, the script looks for naturo.png.

Run the Detection Script Run the run.py file:

python run.py

View the Output

The console will display the total number of detected objects.
The annotated image will be saved in the outputs/ folder (default: outputs/annotated_image.jpg
