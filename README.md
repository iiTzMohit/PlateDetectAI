# PlateDetectAI

## Overview
This project focuses on detecting vehicle number plates using a custom-trained YOLOv3 model. The model has been fine-tuned and optimized for accuracy and performance, specifically targeting number plate detection in various image conditions.

## Features
- **Object Detection**: Uses a YOLOv3 model trained and fine-tuned specifically for number plate detection.
- **Custom Loss Functions**: Utilizes tailored loss functions for better performance.
- **Image Preprocessing**: Includes techniques like cropping, resizing, noise addition, blurring, and rotation.
- **Evaluation Metrics**: Utilizes mAP@[0.50:0.95] and mAP@0.70 for performance evaluation.
- **Efficient Inference**: Benchmarked inference time per image for performance analysis.

## Dataset and Model
You can access the dataset and the saved model used in this project through the following links:
- [Dataset](https://drive.google.com/drive/folders/1h2_CZXVMQYMJoVjeIvmWJc-Hq-hmagZB?usp=sharing)
- [Saved Model](https://drive.google.com/drive/folders/1MaCjaG8O5oxSevaBWByieUxDoZ_2PJr-?usp=sharing)

## Examples
Detected Number Plates in Images

## Model Training Details
- **Custom Loss Functions**: Incorporated custom loss functions tailored for number plate detection.
- **Gradual Layer Unfreezing**: Training strategy with gradual unfreezing of layers.
- **Image Augmentation**: Applied techniques like cropping, noise addition, blurring, and rotation to enhance the dataset.

## Evaluation Metrics
- **Mean Average Precision (mAP)**: Calculated **mAP@[0.5:0.95] of 0.65** and **mAP@0.75 of 0.84** to evaluate model performance.
- **Inference Time**: Benchmarked the model with an average inference time at **t ms/image**.

## Acknowledgements
- The YOLOv3 implementation is inspired by the official YOLOv3 paper and various open-source contributions.
