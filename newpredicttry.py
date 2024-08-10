from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS if needed for cross-origin requests

# Load the saved models
tomato_model = tf.keras.models.load_model('Tomato_Disease_Model.h5')

# Define the class names for the models
tomato_classes = [
    'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite'
]


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(model, class_names, img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]
    return predicted_label

@app.route('/predict/tomato', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    crop_type = 'tomato'

    if not crop_type:
        return jsonify({'error': 'No crop type provided'}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Select the model based on the crop type
    if crop_type == 'tomato':
        model = tomato_model
        class_names = tomato_classes
    else:
        return jsonify({'error': 'Invalid crop type'}), 400

    # Predict the disease
    predicted_disease = predict_disease(model, class_names, file_path)

    # Clean up: remove the file after processing
    os.remove(file_path)

    return jsonify({'predicted_disease': predicted_disease})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
