import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# YOUR MODEL HAS ONLY 3 CLASSES
class_names = ["GLIOMA", "MENINGIOMA", "PITUITARY"]

model = tf.keras.models.load_model(r"C:\Users\guruprasad\Desktop\MRI\models\brain_tumor_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    accuracy = np.max(prediction) * 100
    return result, accuracy
