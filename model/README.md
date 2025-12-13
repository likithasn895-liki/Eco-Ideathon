# Model Folder

This folder will contain the trained solar rooftop detection model (solar_model.pt).
# Solar Panel Detection Model

This directory contains the pre-trained YOLO model for solar panel detection, trained using Roboflow.

## Model Architecture
* YOLOv8n
* Input size: 640x640
* Output: Bounding box coordinates and class probabilities

## Training
* Trained on: [Dataset name] using Roboflow
* Training time: ~2 hours
* Trained using Roboflow's automated training pipeline

## Usage
* Load the model using YOLO("model/best.pt")
* Use the inference.py script to detect solar panels in images
