## Title:
## Breast-Cancer-Detection-Using-YOLOv8-and-Transfer-Learning
![image](https://github.com/user-attachments/assets/2b9c7517-c196-42be-b6a0-159aef6e534e)


## Project Description
This project aims to detect breast cancer using a combination of traditional image processing techniques and modern deep learning models such as VGG16, ResNet50, EfficientNet, and YOLOv8. The dataset used for this project is the CBIS-DDSM dataset. The project performs data preprocessing, model training, and evaluation for image classification and object detection.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [YOLOv8 Training](#yolov8-training)
- [Usage](#usage)
- [Results](#results)

## Features
- Image Preprocessing (CLAHE, Haze Reduction, Grayscale Conversion)
- CNN model training (VGG16, ResNet50, EfficientNet)
- YOLOv8-based object detection for ROI (Regions of Interest)
- Model evaluation with accuracy, precision, recall, and F1 score
- Visualization of image results, confusion matrix, and performance metrics

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Malik329513/Breast-Cancer-Detection-Using-YOLOv8-and-Transfer-Learning-.git
   cd breast-cancer-detection

2. Create and activate a virtual environment
   ```bash
   python -m venv env
   source env/bin/activate 

3. Install dependencies
   ```bash
   pip install -r requirements.txt

## Dataset
The project uses the [CBIS-DDSM Breast Cancer Image Dataset](https://www.cancerimagingarchive.net). It contains full mammogram images, cropped images, and ROI mask images. The dataset includes information on pathology, calcification types, and other metadata.

The dataset is divided into the following:
- Training set: `mass_case_description_train_set.csv`
- Test set: `mass_case_description_test_set.csv`

## Preprocessing
Several preprocessing techniques are applied to the images, including:
- **Grayscale Conversion**: Converts the images to grayscale for simplicity.
- **CLAHE**: Applies Contrast Limited Adaptive Histogram Equalization to improve image contrast.
- **Haze Reduction**: Reduces haze in the images using local and global techniques.
- **ROI Overlay**: Processes images to overlay ROI (Region of Interest) masks on full mammogram images.

## Model Training
The following CNN architectures were trained on the dataset:
- **VGG16**: A deep convolutional network used for image classification.
- **ResNet50**: A residual neural network with 50 layers.
- **EfficientNet**: A more efficient model with fewer parameters for high performance.

The models were trained using data augmentation techniques, early stopping, and model checkpoints.

## YOLOv8 Training
YOLOv8 was used for object detection to identify specific regions of interest (ROI) in the mammogram images. The model was trained using the following steps:
1. Download the pre-trained weights for YOLOv8.
2. Train the model on the labeled dataset for object detection.

   ```bash
   # Command to start training
   train_data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 lr0=0.01

## Usage
To run the project and make predictions:
1. **Train the model**:
   ```bash
   python main.py

## Results
The following results were obtained using different CNN architectures for breast cancer detection:

| CNN Architecture   | Overall Accuracy | Precision | Recall  | F1 Score |
|--------------------|------------------|-----------|---------|----------|
| VGG16              | 59.79%           | 61.20%    | 60.01%  | 60.00%   |
| ResNet50 (TL)      | 67.00%           | 67.21%    | 67.00%  | 67.00%   |
| EfficientNetB0 (TL)| 65.87%           | 65.10%    | 66.04%  | 65.00%   |

**Summary**:
- **VGG16**: Achieved an overall accuracy of 59.79%, with a precision of 61.20% and F1 score of 60.00%.
- **ResNet50 (Transfer Learning)**: Performed better, with an accuracy of 67.00% and a balanced precision, recall, and F1 score.
- **EfficientNetB0 (Transfer Learning)**: Achieved 65.87% accuracy, with balanced metrics in precision and recall.


**YOLOv8:**

| CNN Architecture   | Overall Accuracy | Precision | Recall  | F1 Score |
|--------------------|------------------|-----------|---------|----------|
| yolov8             | 55.47%           | 49.72%    | 55.47%  | 52.34%   |

- Achieved 55.47% accuracy, with a lower precision (49.72%) and F1 score (52.34%), but showed reasonable recall for object detection tasks.

**YOLOv8** successfully detected ROIs.
  
![image](https://github.com/user-attachments/assets/967f7ec8-856e-49f1-a71f-97192ed5615a)

