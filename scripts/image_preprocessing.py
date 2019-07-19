import os
import cv2
import sys
import imutils

import numpy as np 

from tqdm import tqdm


def img_process(img_path, rows, cols, angle=0, pad=20):
    
    img = cv2.imread(img_path, 0)
    
    thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = np.pad(thresh, ((pad,pad),(pad,pad)), 'constant', constant_values=(255,255))

    (len_x, len_y) = thresh.shape
    
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.rotate(thresh, angle)
    thresh = cv2.bitwise_not(thresh)
    
    min_y, _ = np.unravel_index(np.argmin(thresh), (len_x, len_y))
    min_x, _ = np.unravel_index(np.argmin(thresh.T), (len_y, len_x))
    max_y, _ = np.unravel_index(np.argmin(thresh[::-1]), (len_x, len_y))
    max_x, _ = np.unravel_index(np.argmin(thresh.T[::-1]), (len_y, len_x))
    
    max_x = len_x - max_x
    max_y = len_y - max_y

    if min_y > 10:
        min_y -= 10
    if min_x > 10:
        min_x -= 10

    if max_x + 10 < len_x:
        max_x += 10
    if max_y + 10 < len_y:
        max_y += 10

    thresh = thresh[min_y:max_y, min_x:max_x]

    return cv2.resize(thresh, (rows,cols))

def assigner(x, y, counter, path, rows, cols, char, angle=0):

    x[counter, :, :, 0] = img_process(path, rows, cols, angle)
    y[counter] = char

    return counter + 1
    
def main():

    m = 0
    rows, cols = 32, 32

    rotations = 3

    one_char_size = 50
    cv_size = one_char_size * 32 * rotations
    test_size = one_char_size * 32 * rotations

    angles = [0, 15, 345]

    origin_folder = "../data/preprocessed_data"
    destination_folder = "../data/final_data"


    for i in os.listdir(origin_folder):
        if i.isdigit():
            if int(i) > 100:
                m += len(os.listdir(os.path.join(origin_folder, i)))

    train_size = m * rotations - test_size - cv_size 
    
    train_x = np.zeros((train_size, rows, cols, 1), dtype=np.uint8)
    train_y = np.zeros((train_size), dtype=np.uint8)

    test_x = np.zeros((test_size, rows, cols, 1), dtype=np.uint8)
    test_y = np.zeros((test_size), dtype=np.uint8)

    cv_x = np.zeros((cv_size, rows, cols, 1), dtype=np.uint8)
    cv_y = np.zeros((cv_size), dtype=np.uint8)

    counter_train = 0
    counter_test = 0
    counter_cv = 0

    for i in os.listdir(origin_folder):

        if i.isdigit():
            int_i = int(i)
            char = int_i - 1072

            if int_i > 100:
                img_path = os.path.join(origin_folder, i)
                len_folder = len(os.listdir(img_path))
                len_folder_train = len_folder - one_char_size

                for k, j in tqdm(enumerate(os.listdir(img_path))):
                    
                    path_to_img = os.path.join(img_path, j)
                    
                    if k < one_char_size:
                        
                        for i in range(rotations):
                            counter_test = assigner(test_x, test_y, counter_test, path_to_img, rows, cols, char, angles[i])

                    elif k >= one_char_size and k < one_char_size * 2:

                        for i in range(rotations):
                            counter_cv = assigner(cv_x, cv_y, counter_cv, path_to_img, rows, cols, char, angles[i])
                    
                    else:
                        
                        for i in range(rotations):
                            counter_train = assigner(train_x, train_y, counter_train, path_to_img, rows, cols, char, angles[i])

    print(train_y[-1])
    print(test_y[-1])
    print(cv_y[-1])

    np.save(os.path.join(destination_folder, f'data_{rows}x{cols}_size={m}_rotations={rotations}_cv.npy'), [train_x, train_y, test_x, test_y, cv_x, cv_y])

    print('train shape =', train_x.shape)
    print('test shape =', test_x.shape)
    print('cv shape =', cv_x.shape)

if __name__ == '__main__':
    main()