import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import copy as cp
import pathlib
from tensorflow.keras import layers

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Concatenate

#################데이터 불러오기####################
data_dir = pathlib.Path('./space_data')
data_dir

batch_size = 16
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

################데이터 증강####################
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

#################모델 생성#####################
from keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPooling2D , Dropout, Flatten


base_model = VGG19(input_shape=(img_height, img_width , 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

from keras.layers import Dense , GlobalAveragePooling2D

global_average_layer = GlobalAveragePooling2D()
# prediction_layer = Dense(100)



VGGmodel2 = tf.keras.Sequential([
  data_augmentation,
  base_model,
  global_average_layer,
  Flatten(),
  Dense(64,activation="relu"),
  Dense(16,activation="relu"),
  Dense(len(class_names), activation="softmax")
])


VGGmodel2.compile(loss="sparse_categorical_crossentropy",optimizer = 'adam', metrics=["accuracy"])

from keras.callbacks import EarlyStopping , ReduceLROnPlateau

learning_rate_reduction=ReduceLROnPlateau(
                        monitor= "val_loss",
                        patience = 3,
                        factor = 0.5,
                        min_lr=0.0001,
                        verbose=1)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 학습이 언제 자동 중단 될지를 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

#최적화 모델이 저장될 폴더와 모델의 이름을 정합니다.
modelpath="./model/space_classification_model.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)


######################모델 학습###############################
history= VGGmodel2.fit(train_ds,
                    validation_data=val_ds,
                    epochs=20,
                      )

VGGmodel2.evaluate(val_ds)
# 손실값, 정확도