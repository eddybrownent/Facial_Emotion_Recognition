# Facial Emotion Recognition Project

## Description

This project focuses on building a Convolutional Neural Network (CNN) for Facial Emotion Recognition. 
The goal is to train a model that can accurately classify facial expressions into predefined emotion categories. 
The dataset used for training includes images with labeled emotions, and the model is implemented using the TensorFlow and Keras libraries.

## How to Run

Follow the steps below to run the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/facial-emotion-recognition.git
   cd facial-emotion-recognition

## Install Dependencies:
Ensure that you have the required dependencies installed. You can use the following command:

pip install -r requirements.txt

## Download Dataset:
Download the facial emotion dataset and place it in the appropriate directory:
wget https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/download?datasetVersionNumber=1

## Train the Model:
Train the Facial Emotion Recognition model using the provided script:

python train_model.py


## Evaluate the Model:
Evaluate the trained model on the test set:

python evaluate_model.py

## Predict on New Image:
Use the trained model to predict the emotion in a new image:

python predict_image.py path/to/your/image.jpg
