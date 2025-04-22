from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 160

# Load saved model
model = keras.models.load_model('cats_vs_dogs_model.h5')

# Load image
img_path = 'PersonTest.jpg'
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

# Visualize Prediction
categories = ['Cats', 'Dogs']
value = float(prediction[0][0])
cat = 1 - value
dog = value
values = [cat, dog]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)

plt.figure(figsize=(6,4))
plt.pie(values, labels=categories, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Model Prediction Breakdown')
plt.axis('equal')
plt.show()