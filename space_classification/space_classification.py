import json
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import urllib.request
import time
import collections

def img_download(url):
    # time check
    start = time.time()
    img_path_list = []
    for cnt, i in enumerate(url):
        img_path = f"./space_classification/img/test{cnt+1}.jpg"
        # 이미지 요청 및 다운로드
        urllib.request.urlretrieve(i, img_path)
        img_path_list.append(img_path)
    # 이미지 다운로드 시간 체크
    print(time.time() - start)

    return img_path_list

def space_classification(image_path, classification_model):
    best_score = []
    best_class = []
    with open("./space_classification/space_class_names.json", "r") as f:
        data = json.load(f)
        class_names = []
    for i in image_path:

        class_names = data['space_class_names']
        img = keras.preprocessing.image.load_img(
            i, target_size=(224, 224)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = classification_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        #class , score 각각 list로 저장
        best_class.append(np.argmax(score))
        best_score.append(score[np.argmax(score)].numpy())

    # 사진이 여러개 일경우
    if len(best_class) >= 2:

        #최빈값 비교
        counts = collections.Counter(best_class)
        #최빈값 중복시 평균값 비교후 높은값 추론
        counts_most_two = counts.most_common(2)
        if counts_most_two[0][1] == counts_most_two[1][1]:
            most_two = {counts_most_two[0][0]: [], counts_most_two[0][1]: []}
            for cnt, x in enumerate(best_class):
                if counts_most_two[0][0] == x:
                    most_two[counts_most_two[0][0]].append(cnt)
                else:
                    most_two[counts_most_two[1][0]].append(cnt)
            most1 = []
            most2 = []
            for x in most_two[counts_most_two[0][0]]:
                most1.append(best_score[x])
            for y in most_two[counts_most_two[1][0]]:
                most2.append(best_score[y])
            if np.mean(most1) > np.mean(most2):
                final_class = counts_most_two[0][0]
            else:
                final_class = counts_most_two[1][0]

    else:
        counts = collections.Counter(best_class)
        final_class = counts.most_common(1)[0][0]
    return class_names[final_class]
