import os
import cv2
import sys

import numpy as np

from PIL import Image
from tqdm import tqdm

origin_folder = "../data/raw_data"
destination_folder = "../data/preprocessed_data"

counter = 0
size = 0

def progressbar(counter, size, aggregator='=', progressbar_length=50):
	percent = counter/size
	progress = int(progressbar_length * percent)
	left = int(progressbar_length * (1 - percent))
	print('\r',end='')
	print(f"{int(percent*100)}% |{'='*(progress-1)}>{'.'*left}| {counter}/{size}", end='')
	if percent == 1:
		print('\r',end='')
		print(f"{int(percent*100)}% |{'='*(progress)}{'.'*left}| {counter}/{size}", end='')


for folder in os.listdir(origin_folder):
	
	temp_path = os.path.join(origin_folder, folder)

	for char_folder in os.listdir(temp_path):
		f_path = os.path.join(destination_folder, str(ord(char_folder.lower())))
		
		if not os.path.isdir(f_path):
			os.mkdir(f_path)
		
		size += len(os.listdir(os.path.join(temp_path, char_folder)))

for folder in os.listdir(origin_folder):
	
	temp_path = os.path.join(origin_folder, folder)

	for char_folder in os.listdir(temp_path):

		for i in os.listdir(os.path.join(temp_path, char_folder)):
			if i.endswith('.png'):
				path_to_image = os.path.join(temp_path, char_folder, i)
				
				img = Image.open(path_to_image)
					
				img.save(f"{os.path.join(destination_folder, str(ord(char_folder.lower())))}/{ord(char_folder.lower())}_{counter}.jpg")
			
				counter+=1
				if counter % 100 ==0:
					progressbar(counter, size)