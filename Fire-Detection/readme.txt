Fire-Detection
Overview
Fire-Detection is a robust AI-powered solution designed for real-time fire detection using advanced computer vision techniques. This repository utilizes the YOLO (You Only Look Once) algorithm to accurately identify fire in various environments. The project aims to provide a reliable method for early fire detection to enhance safety and prevent fire-related disasters.

Features
Real-time Fire Detection: Leverages YOLO for high-speed, accurate fire detection.
Customizable Models: Adaptable to different environments and fire scenarios.
Easy Integration: Compatible with various systems for seamless deployment.
Alert System: Configurable to trigger alerts for immediate action.
Open Source: Encourages community contributions and improvements.
Installation
To set up the Fire-Detection project, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/Dan-445/ultralytics.git
Navigate to the project directory:

bash
Copy code
cd ultralytics/Fire-Detection
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the YOLOv8 model:

bash
Copy code
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8n.pt
Usage
You can use Fire-Detection both from the command line and within a Python environment.

Command Line Interface (CLI)
Run the YOLO model on an input video:

bash
Copy code
yolo predict model=yolov8n.pt source='path_to_video.mp4'
Python
Use the model in a Python script:

python
Copy code
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('path_to_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Visualize results
    annotated_frame = results[0].plot()
    cv2.imshow('Fire Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Model Training
To train the YOLO model on a custom dataset, follow the instructions below:

Prepare your dataset: Ensure your dataset is annotated and organized in YOLO format.
Modify the configuration: Adjust the model and data configurations as required.
Train the model:
bash
Copy code
yolo train model=yolov8n.yaml data=your_dataset.yaml epochs=50
Contributing
Contributions to the Fire-Detection project are welcome! Feel free to open issues, submit pull requests, or suggest new features.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Special thanks to the Ultralytics team for their continuous efforts in developing and maintaining the YOLO series. For more details and resources, visit the Ultralytics GitHub page.
