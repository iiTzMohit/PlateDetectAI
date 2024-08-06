# PlateDetectAI

## Overview
This project focuses on detecting vehicle number plates using a custom-trained YOLOv3 model. The model has been fine-tuned and optimized for accuracy, specifically targeting number plate detection in various image conditions.

## Features
- **Object Detection**: Uses a YOLOv3 model trained and fine-tuned specifically for number plate detection.
- **Custom Loss Functions**: Utilizes tailored loss functions for better performance.
- **Image Preprocessing**: Includes techniques like cropping, resizing, noise addition, blurring, and rotation.
- **Evaluation Metrics**: Utilizes mAP@[0.50:0.95] and mAP@0.70 for performance evaluation.

## Dataset and Model
You can access the dataset and the saved model used in this project through the following links:
- [Dataset](https://drive.google.com/drive/folders/1h2_CZXVMQYMJoVjeIvmWJc-Hq-hmagZB?usp=sharing)
- [Saved Model](https://drive.google.com/drive/folders/1MaCjaG8O5oxSevaBWByieUxDoZ_2PJr-?usp=sharing)

## Examples
Detected Number Plates in Images

![image](https://github.com/user-attachments/assets/798c6b9d-94d4-4e2d-b694-3b58c79df9b4)

![image](https://github.com/user-attachments/assets/808a2456-a753-4f3c-b1ad-b73024976b6f)


## Model Training Details
- **Custom Loss Functions**: Incorporated custom loss functions tailored for number plate detection.
- **Gradual Layer Unfreezing**: Training strategy with gradual unfreezing of layers.
- **Image Augmentation**: Applied techniques like cropping, noise addition, blurring, and rotation to enhance the dataset.

## Run Inference
To run inference on new images, open the provided Jupyter notebook `detect.ipynb`. This notebook contains the code to perform inference and visualize the results.

## Acknowledgements
The YOLOv3 implementation is inspired by the official YOLOv3 paper and various open-source contributions.
