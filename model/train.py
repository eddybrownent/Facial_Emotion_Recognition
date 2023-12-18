import pandas as pd
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image


# Load and preprocess data, define data generators, etc.
data = pd.read_csv('data.csv')

# Define a generator function to load and preprocess images in batches
def data_generator(data, batch_size=32):
    num_samples = len(data)
    while True:
        indices = np.random.choice(num_samples, batch_size, replace=False)
        X_batch = np.array([img_to_array(load_img('/home/edwin/Facial_Emotion_Recognition/dataset/' + img, target_size=(224, 224))) for img in data['path'].iloc[indices]])
        y_batch = LabelEncoder().fit_transform(data['label'].iloc[indices])
        yield X_batch, y_batch


label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['label'])


# Convert the encoded labels to one-hot encoding
one_hot_labels = to_categorical(data['encoded_label'], num_classes=6)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


# Create an image data generator with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Create an image data generator without augmentation for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create an image data generator without augmentation for testing
test_datagen = ImageDataGenerator(rescale=1./255)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

sample_generator = data_generator(train_data, batch_size=32)
X_sample, y_sample = next(sample_generator)

# If your labels are not one-hot encoded, you can directly use them
# Convert categorical labels to numerical indices
y_sample_indices = y_sample

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_sample_indices), y=y_sample_indices)

# Create a dictionary to be used as class_weight parameter in the generator
class_weight_dict = dict(enumerate(class_weights))


batch_size = 8
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='/home/edwin/Facial_Emotion_Recognition/dataset/',
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    class_weight=class_weight_dict
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory='/home/edwin/Facial_Emotion_Recognition/dataset/',
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory='/home/edwin/Facial_Emotion_Recognition/dataset/',
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained ResNet50 model without including the top (fully connected) layers
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# fine-tuning unfreezing the last few layers of the base model
for layer in base_model.layers[-10:]:
    layer.trainable = False

# Create model on top of the base model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history_fine_tuned = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Save the trained model
model.save('trained_model.h5')

# Evaluate the model on the test set
score = model.evaluate(test_generator, steps=len(test_generator), verbose=0)

# Print the test loss and accuracy
print('Test Loss =', score[0])
print('Test Accuracy =', score[1])

# Evaluate the model on the test set
score = model.evaluate(test_generator, steps=len(test_generator), verbose=0)

# Print the test loss and accuracy
print('Test Loss =', score[0])
print('Test Accuracy =', score[1])

# Predict the classes for the test set
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator, steps=len(test_generator))
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate precision, recall, and F1 score
from sklearn.metrics import classification_report

class_names = list(test_generator.class_indices.keys())

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Evaluate the model on the validation set
val_score = model.evaluate(val_generator, steps=len(val_generator), verbose=0)

# Print the validation loss and accuracy
print('Validation Loss =', val_score[0])
print('Validation Accuracy =', val_score[1])

# Predict the classes for the validation set
y_val_true = val_generator.classes
y_val_pred_prob = model.predict(val_generator, steps=len(val_generator))
y_val_pred = np.argmax(y_val_pred_prob, axis=1)

# Print classification report for the validation set
print("\nValidation Classification Report:")
print(classification_report(y_val_true, y_val_pred, target_names=class_names))
