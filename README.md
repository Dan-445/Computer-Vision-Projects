# AI Computer Vision Projects

Welcome to the **AI Computer Vision Projects** repository! This collection showcases a variety of AI-driven computer vision models and solutions, ranging from object detection to specific applications like garbage detection and sign language recognition. Each project leverages state-of-the-art deep learning models, including YOLOv5, YOLOv8, and Detectron2.

---

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Project Overviews](#project-overviews)
  - [Garbage Detection (YOLOv5)](#garbage-detection-yolov5)
  - [Object Detection (Ubuntu)](#object-detection-ubuntu)
  - [Plant Disease Detection](#plant-disease-detection)
  - [Rotten & Fresh Fruits Detection](#rotten--fresh-fruits-detection)
  - [SCARLET-TANAGER-EYE](#scarlet-tanager-eye)
  - [Turkish Sign Language Deployment (Flask + YOLOv8)](#turkish-sign-language-deployment-flask--yolov8)
  - [NSFW Detection (YOLOv5)](#nsfw-detection-yolov5)
  - [Pothole Detection (Custom Detectron2)](#pothole-detection-custom-detectron2)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

This repository contains a set of AI projects focused on solving various computer vision challenges. Whether you're detecting potholes, recognizing sign language, or classifying fresh vs. rotten fruits, these projects showcase the power and versatility of AI models in different domains. Each project is self-contained and demonstrates practical AI solutions that can be adapted or extended based on your needs.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- [PyTorch](https://pytorch.org/get-started/locally/)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- Flask (for Turkish Sign Language Deployment)
- OpenCV
- CUDA (Optional for GPU acceleration)

To install dependencies, run:

```bash
git clone https://github.com/your-username/ai-computer-vision-projects.git
cd ai-computer-vision-projects
pip install -r requirements.txt
Project Overviews
1. Garbage Detection (YOLOv5)
Detect and classify various types of garbage in images to help environmental agencies or smart waste management systems. The model can distinguish between different types of waste, such as plastic, glass, and metal.

Model: YOLOv5
Dataset: Custom labeled dataset for garbage detection.
Applications: Waste sorting, environmental monitoring.
2. Object Detection (Ubuntu)
A versatile object detection system configured specifically for Ubuntu environments. This project demonstrates how to set up and deploy object detection models using YOLO and OpenCV on Ubuntu systems.

Model: YOLOv5/YOLOv8
Environment: Ubuntu
Applications: General object detection, surveillance, robotics.
3. Plant Disease Detection
A deep learning model to identify and classify plant diseases based on leaf images. This model helps farmers and agricultural experts quickly diagnose issues in crops.

Model: CNNs
Dataset: Plant Village dataset.
Applications: Precision agriculture, plant health monitoring.
4. Rotten & Fresh Fruits Detection
A computer vision model to classify fruits as either fresh or rotten, aiding in food quality control systems.

Model: CNN/YOLO
Dataset: Custom dataset of fruits.
Applications: Food industry, supermarkets, food processing units.
5. SCARLET-TANAGER-EYE
This project simulates the visual perception of the Scarlet Tanager bird, focusing on its unique color detection abilities.

Model: Custom Vision Model
Applications: Ecology, bird vision studies, biology research.
6. Turkish Sign Language Deployment (Flask + YOLOv8)
A web-based application powered by Flask and YOLOv8 to recognize and translate Turkish sign language gestures.

Model: YOLOv8
Framework: Flask for deployment.
Applications: Communication for the hearing-impaired, sign language recognition.
7. NSFW Detection (YOLOv5)
A machine learning model to detect NSFW (Not Safe for Work) content. This can be used for content moderation in online platforms.

Model: YOLOv5
Dataset: Public NSFW dataset.
Applications: Content moderation, social media platforms, image filtering.
8. Pothole Detection (Custom Detectron2)
A custom implementation of Detectron2 to identify potholes in road images, assisting in road maintenance and safety.

Model: Custom Detectron2 implementation.
Dataset: Pothole images dataset.
Applications: Autonomous driving, road safety, urban planning.
Usage
To run any of the projects:

Navigate to the specific project folder (e.g., garbage-detection/).
Follow the setup instructions in each project folder's README to load the dataset and model weights.
Run the Python script associated with the project:
bash
Copy code
python detect.py --source <path_to_images> --weights <path_to_weights>
Make sure to adjust paths and configurations according to your setup.

Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix: git checkout -b feature-name.
Commit your changes: git commit -m 'Add some feature'.
Push to the branch: git push origin feature-name.
Submit a pull request.
License
This repository is licensed under the MIT License. See the LICENSE file for details.
