import os
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def space_classification(image):
    path = './space_data'
    class_names = os.listdir(path)

    classification_model = tf.keras.models.load_model('./space_classification_model/space_classification_model.hdf5')
    img = keras.preprocessing.image.load_img(
        image, target_size=(224, 224)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = classification_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return print(class_names[np.argmax(score)])

