from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

# Load trained model
model = tf.keras.models.load_model("cow_skin_disease_model.h5")

# Disease classes
classes = ['foot-and-mouth', 'healthy', 'lumpy']

# Upload folder
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(file_path)

    # Load image
    img = image.load_img(file_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = float(np.max(prediction) * 100)

    return jsonify({
        'disease': predicted_class,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)