from tkinter import *

import cv2
import win32gui

import numpy as np

from time import time
from PIL import ImageGrab
from keras.models import load_model

np.set_printoptions(suppress=True)

canvas_width = 500
canvas_height = 500
history = {}

path_to_model = "../models/charRecogn_32x32_(3200, 32, 32, 1).h5"

model = load_model(path_to_model)

elements = []

kernel = np.ones((5,5),np.uint8)

def detect(rect):

    img = ImageGrab.grab(rect)

    width = img.width
    height = img.height
    raw = np.frombuffer(img.tobytes(), dtype=np.uint8)
    img = raw.reshape((height, width, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.erode(gray, kernel, iterations = 3)

    data = cv2.resize(gray, (32,32))/255.
    
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=-1)

    pred = model.predict(data)

    print(chr(np.argmax(pred)+1072).upper())

def paint(event):
    python_green = "#000000"
    if history and (time() - history['time']) < 0.1:
        elements.append(w.create_line(event.x, event.y, history['x'], history['y'], width=18))
        elements.append(w.create_oval(event.x - 9, event.y - 9, event.x + 9, event.y + 9, fill="black"))
        history['x'] = event.x
        history['y'] = event.y
        history['time'] = time()

    else:
        history['x'] = event.x
        history['y'] = event.y
        history['time'] = time()

master = Tk()
master.title( "Painting using Ovals" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

HWND = w.winfo_id()

button = Button(master, text="распознать", command=lambda x = HWND: detect(win32gui.GetWindowRect(x)))
button.pack()

button1 = Button(master, text="очистить", command=lambda : [w.delete(i) for i in elements])
button1.pack()

message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )

    
mainloop() 