# app.py (Flask web service with class mapping)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('your_updated_model.h5')

# Define class mapping
class_mapping = {0: "Ahegao", 1: "Angry", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprise"}

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess the input image
    img_file = request.files['image']
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values consistently with training

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming it's a classification task, get the predicted class index
    predicted_class = np.argmax(predictions)

    # Map the class index to the corresponding label using class_mapping
    predicted_label = class_mapping.get(predicted_class, "Unknown")

    return jsonify({'predicted_class': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
