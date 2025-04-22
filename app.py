from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = keras.models.load_model('cats_vs_dogs_model.h5')
IMG_SIZE = 160

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            result = model.predict(img_array)
            prediction = "🐶 Dog" if result[0][0] > 0.5 else "😺 Cat"

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)