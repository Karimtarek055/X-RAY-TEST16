from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

model = load_model(r'model\keras_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        prediction = classify_image(file)

        return render_template('result.html', filename=file.filename, prediction=prediction)
    print(request.url)
    return redirect(request.url)


def classify_image(file):
    img = Image.open(file.stream)  # Open the image file from the file object's stream
    img = img.convert('RGB')  # Convert image to RGB format
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_names = open("model/labels.txt", "r").readlines()
    class_label = f'Class is: {class_names[class_idx]}'
    return class_label


if __name__ == '__main__':
    app.run(debug=True)
