# Brain Tumor Classification with VGG16

## Overview

This project classifies brain MRI images to detect tumors using VGG16, a pre-trained Convolutional Neural Network (CNN). It includes preprocessing, data augmentation, and training using a binary classification model.

## Dataset

The dataset contains MRI images categorized into two classes: `YES` (tumor) and `NO` (healthy). The data is split into `train`, `val`, and `test` directories.

## Features

* **Preprocessing**: Images resized to 224x224 and normalized using VGG16's preprocessing function.
* **Data Augmentation**: Includes random rotation, shift, and brightness adjustment.
* **Model**: VGG16 base with custom fully connected layers for binary classification.
* **Training**: Adam optimizer with binary cross-entropy loss.

## Installation

```bash
pip install imutils opencv-python plotly keras scikit-learn matplotlib
```

## Code Breakdown

1. **Data Loading**: Images are loaded and preprocessed for VGG16.
2. **Model Architecture**: VGG16 is fine-tuned with added fully connected layers.
3. **Training**: Model is trained using augmented data.
4. **Evaluation**: Model accuracy and confusion matrix calculated for validation and test sets.

## Results

* **Validation Accuracy**: 90.0%
* **Test Accuracy**: 100.0%

## Usage

To predict the class of a new image:

```python
img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

pred = model.predict(img_array)
print("Predicted:", "Yes (Tumor)" if pred > 0.5 else "No (Healthy)")
```

