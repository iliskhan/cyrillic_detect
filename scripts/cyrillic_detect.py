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
    model_folder = r"C:\Users\020GudaevII\Desktop\dev\cyrillic_detect\models"

    train_x, train_y, test_x, test_y, cv_x, cv_y = np.load(os.path.join(data_folder, 'data_32x32_rotated=1_cv.npy'), allow_pickle=True)

    m_train, m_test = train_x.shape[0], test_x.shape[0]

    rows, cols = train_x.shape[1:3]

    train_x = train_x / 255.
    test_x = test_x / 255.
    cv_x = cv_x / 255.

    batch_size = 32

    regularizer = regularizers.l2(0.001)

    C = tf.constant(32, name="C")

    train_y, test_y, cv_y = [tf.one_hot(i,C,axis=1) for i in (train_y, test_y, cv_y)]

    with tf.Session() as sess:
        train_y, test_y, cv_y = sess.run([train_y,test_y, cv_y])

    X_input = Input((rows, cols, 1))

    #X = Dropout(0.9)(X_input)
    
    # Zero-Padding: pads the border of X_input with zeroes
    #X = ZeroPadding2D((3, 3))(X_input)


    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0', padding='same')(X_input)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1', padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv2', padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    X = Flatten()(X)

    X = Dense(1024, activation='relu', name='fc0')(X)
    X = Dropout(0.2)(X)

    X = Dense(512, activation='relu', name='fc1')(X)
    X = Dropout(0.2)(X)
        
    X = Dense(32, activation='softmax', name='fc2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='charRecogn') 

    model.compile(optimizer="Adam",loss="categorical_crossentropy", metrics=["accuracy"])

    char_callback = keras.callbacks.TensorBoard(log_dir=f'../logs/{rows}x{cols}_{train_x.shape}', batch_size=batch_size, write_images=True, update_freq=5000)
    
    model.fit(x=train_x, y=train_y, epochs=20, batch_size=batch_size, validation_data=(cv_x, cv_y), callbacks=[char_callback])

    preds = model.evaluate(x=test_x, y=test_y)

    print("Loss =", preds[0])
    print("Test accuracy =", preds[1])

    model.save(f'{model_folder}/charRecogn_{rows}x{cols}_{test_x.shape}.h5')

    keras.utils.print_summary(model)

if __name__ == '__main__':
    main()
