import os
import cv2
import keras

import numpy as np 
import tensorflow as tf 

from keras import backend as K
from keras import regularizers

from keras.models import Model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D


def main():

    data_folder = "../data/final_data"
    model_folder = "../models"
    data_name = "data_32x32_size=86214_cv.npy"

    train_x, train_y, test_x, test_y, cv_x, cv_y = np.load(os.path.join(data_folder, data_name), allow_pickle=True)

    m_train, m_test = train_x.shape[0], test_x.shape[0]

    rows, cols = train_x.shape[1:3]

    train_x = train_x / 255.
    test_x = test_x / 255.
    cv_x = cv_x / 255.

    batch_size = 32

    regularizer = regularizers.l2(0.001)

    C = tf.constant(32, name="C")

    train_y, test_y, cv_y = [tf.expand_dims(tf.expand_dims(tf.one_hot(i,C,axis=1), 1), 1) for i in (train_y, test_y, cv_y)]

    with tf.Session() as sess:
        train_y, test_y, cv_y = sess.run([train_y, test_y, cv_y])
    
    X_input = Input((rows, cols, 1))

    #X = Dropout(0.9)(X_input)

    #X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    X = Dropout(0.2)(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Dropout(0.2)(X)

    X = Conv2D(256, (6, 6), name='fc0')(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    X = Conv2D(256, (1, 1), name='fc1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)

    X = Conv2D(32, (1, 1), name='fc2')(X)
    X = Activation('softmax')(X)
    # X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2', padding='same')(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((2, 2), name='max_pool2')(X)
    # X = Dropout(0.2)(X)
    
    # X = Flatten()(X)

    # X = Dense(256, activation='tanh', name='fc0')(X)
    # X = Dropout(0.4)(X)

    # X = Dense(256, activation='tanh', name='fc1')(X)
    # X = Dropout(0.4)(X)
        
    # X = Dense(32, activation='softmax', name='fc2')(X)

    model = Model(inputs = X_input, outputs = X, name='charRecogn') 

    model.compile(optimizer="Nadam",loss="categorical_crossentropy", metrics=["accuracy"])

    char_callback = keras.callbacks.TensorBoard(log_dir=f'../logs/{rows}x{cols}_{train_x.shape}', batch_size=batch_size, write_images=True, update_freq=5000)
    
    model.summary()
    
    model.fit(x=train_x, y=train_y, epochs=15, batch_size=batch_size, validation_data=(cv_x, cv_y), callbacks=[char_callback])

    preds = model.evaluate(x=test_x, y=test_y)

    print("Loss =", preds[0])
    print("Test accuracy =", preds[1])

    model.save(f'{model_folder}/charRecogn_{rows}x{cols}_{test_x.shape}.h5')

    

if __name__ == '__main__':
    main()
