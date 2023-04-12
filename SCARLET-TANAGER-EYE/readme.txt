
### Scarlet Tanager Eye Detection Model using YOLOv7
This project is an implementation of an eye detection model using YOLOv7 architecture. The model is trained to detect eyes of Scarlet Tanager birds in images.

### Installation
Clone this repository: 
Install the required packages: pip install -r requirements.txt

### Usage
Navigate to the project directory: cd scarlet-tanager-eye-detection-yolov7
Run the detect.py script to detect eyes in an image: python detect.py --image path/to/image.jpg
The script will output an image with bounding boxes around the detected eyes.

### Training
If you wish to train the model on your own dataset, follow these steps:

Collect and label your dataset of Scarlet Tanager bird images.
Create a new YOLOv7 project using the YOLOv7_DeepSort_Pytorch repository (https://github.com/Michael-OvO/YOLOv7_DeepSort_Pytorch).
Replace the data.yaml file in the project with the one provided in this repository.
Place your labeled images in the train/images folder and the corresponding label files in the train/labels folder.
Train the model using the train.py script in the YOLOv7_DeepSort_Pytorch repository.
Once the model is trained, use the detect.py script in this repository to detect eyes in your images.

### Credits
This project was inspired by the YOLOv7 architecture developed by Ultralytics LLC.

The YOLOv7_DeepSort_Pytorch repository (https://github.com/Michael-OvO/YOLOv7_DeepSort_Pytorch) was used as a base for training the eye detection model.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
