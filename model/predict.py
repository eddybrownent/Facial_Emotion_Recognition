import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Load the model with a custom object scope
with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    model = load_model('your_updated_model.h5')


# Load and preprocess the input image
img_path = 'cropped_emotions.6285.png'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))  # Adjust the target_size if needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values consistently with training

# Make predictions
predictions = model.predict(img_array)


# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)
print("Predicted class:", predicted_class)

data = pd.read_csv('data.csv')
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['label'])
one_hot_labels = to_categorical(data['encoded_label'], num_classes=6)

# Map the predicted class to the corresponding one-hot encoded label
predicted_one_hot_label = to_categorical(predicted_class, num_classes=6)

# Reverse the one-hot encoding to get the original encoded label
predicted_encoded_label = np.argmax(predicted_one_hot_label)

# Reverse the LabelEncoder to get the original emotion label
predicted_emotion = label_encoder.inverse_transform([predicted_encoded_label])[0]
print("Predicted emotion:", predicted_emotion)
