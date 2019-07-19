import os
import cv2
import keras

import numpy as np 
import tensorflow as tf 

from keras.models import load_model

path_to_model = "../models/charRecogn_32x32_(3200, 32, 32, 1).h5"
path_to_data = "../data/final_data/data_32x32_size=98100_cv.npy"

train_x, train_y, test_x, test_y, cv_x, cv_y = np.load(path_to_data, allow_pickle=True)

m_train, m_test = train_x.shape[0], test_x.shape[0]

rows, cols = train_x.shape[1:3]

train_x = train_x / 255.
test_x = test_x / 255.
cv_x = cv_x / 255.

C = tf.constant(32, name="C")

train_y, test_y, cv_y = [tf.one_hot(i,C,axis=1) for i in (train_y, test_y, cv_y)]

with tf.Session() as sess:
    train_y, test_y, cv_y = sess.run([train_y,test_y, cv_y])


model = load_model(path_to_model)
for i in train_x:

    img = np.expand_dims(i, axis=0)

    print(chr(np.argmax(model.predict(img))+1072))

    cv2.imshow('img', cv2.resize(i, (70,70)))
    cv2.waitKey()