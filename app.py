from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model('plant_disease_model3.h5')

# Load class labels
with open('model/labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]


# Prediction Function
def predict_disease(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = round(100 * np.max(prediction), 2)

    return predicted_class, confidence


# Home Route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(file_path)

    disease, confidence = predict_disease(file_path)

    return render_template(
        'result.html',
        disease=disease,
        confidence=confidence,
        image_path=file_path
    )


# Run App
if __name__ == '__main__':
    app.run(debug=True)