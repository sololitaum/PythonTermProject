from tensorflow import keras
import numpy as np

IMG_SIZE = 160

# Load saved model
model = keras.models.load_model('cats_vs_dogs_model.h5')

# Load image
img_path = 'DogTest.jpg'
img = keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Print result
if prediction[0] < 0.5:
    print("😺 It's a cat!")
else:
    print("🐶 It's a dog!")
