import json
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import urllib.request
import time

def img_download(url):
    # time check
    start = time.time()
    img_path = "./space_classification/img/test.jpg"
    # 이미지 요청 및 다운로드
    urllib.request.urlretrieve(url, img_path)
    # 이미지 다운로드 시간 체크
    # print(time.time() - start)

    return img_path

def space_classification(image_path, classification_model):

    # class_names = []
    # with open("./space_classification/space_class_names.json", "r") as f:
    #     data = json.load(f)
    # class_names = data['space_class_names']

    img = keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = classification_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    best_score = sorted(score, reverse=True)[0:3]
    best_score = np.array(best_score)
    best_score = list(best_score)

    best_category = np.argsort(score)[:4:-1]
    best_category = list(best_category)

    return best_category, best_score

