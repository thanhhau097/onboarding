import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

# STEP 1: GET AND SAVE CHARACTER IMAGES FROM LINE IMAGES
def read_images(path):
	'''
	Read line images and labels from DIGITAL directory
		:param path: path to data folders
		:return images: line images
		:return labels: each label correspond to each image
	'''
    images = []
    labels = []
    for sub_folder in os.listdir(path):
        sub_path = os.path.join(path, sub_folder, 'images')
        for image_name in os.listdir(sub_path):
            if 'jpg' in image_name:
                image_path = os.path.join(sub_path, image_name)
                image = 255 - cv2.imread(image_path, 0)
                image[image < 30] = 0
                images.append(image)
                labels.append(image_name.split('.')[0])
            else:
                continue
    return images, labels

def get_character_by_connected_components(image):
	'''
	Get characters from input image
		:param image: input image
		:return char_images: character images getting from input image, sorted left to right
	'''
    char_images = []
    
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    chars = list(filter(lambda x: (x[-1] < 2000) and (x[-1] > 100), stats))    
    chars.sort(key=lambda x: x[0])
    
    for i in range(len(chars)):
        left, top, width, height, _ = tuple(chars[i])
        char_images.append(image[top:top+height, left:left+width])
        
    return char_images

def save_characters(images, labels):
	'''
	Read line images and save their characters into character's folders
		:param images: line images 
		:param labels: label correspond to each image
	'''
    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        char_images = get_character_by_connected_components(image)
        for j in range(min(len(label), len(char_images))):
            try:
                char_path = os.path.join('data/characters', str(label[j]), str(i) + '_' + str(j) + '.png')
                cv2.imwrite(char_path, char_images[j]) 
            except:
                break 

# STEP 2: PREPROCESSING & GENERATING LINE IMAGES FROM CHARACTER IMAGESs

def resize_image(image, new_height=64):
	'''
	Resize input image to image with height equal to new_height
		:param image: input image to resize
		:new_height: height of input to resize
		:return resized_image: image after resizing
	'''
    height = image.shape[0]
    width = image.shape[1]
    
    resized_image = cv2.resize(image, (int(new_height*width/height), new_height))
    return resized_image


def get_character_images_from_folder(data_folder):
	'''
	Get images of characters from character folders
		:param data_folder: path to data folders
		:return images: number images
		:return labels: each label correspond to each image
	'''
	images = []
	labels = []

	for sub_folder in os.listdir(data_folder):
	    path = os.path.join(data_folder, sub_folder)
	    
	    # read images in each folder of characters
	    for image_name in os.listdir(path):
	        if 'png' not in image_name:
	            continue
	        image_path = os.path.join(path, image_name)
	        image = 255 - cv2.imread(image_path, 0)
	        image = resize_image(image)
	        if image.shape[1] <= 2:
	            continue
	        
	        images.append(image)
	        labels.append(sub_folder)
        
    return images, labels

def generate(character_folder='data/characters', config_file_folder='dev', output_folder='output_digital'):
	'''
	Using data loader of @john to generate new line images
	'''
	images, labels = get_character_images_from_folder(character_folder)

	# dump data into pickle file 
	with open(os.path.join(config_file_folder, 'digital_numbers.pkl'), 'wb') as f_out:
	    pickle.dump((images, labels), f_out)

	from dataloader.generate.image import HandwrittenLineGenerator, PrintedLineGenerator

	TEST_PKL_FILE = os.path.join(config_file_folder, 'digital_numbers.pkl')
	TEST_TXT_FILE = os.path.join(config_file_folder, 'text_to_gen_digital.txt')
	ALLOWED_CHARS = os.path.join(config_file_folder, 'charset_codes.txt')
	BACKGROUND_NOISES = os.path.join(config_file_folder, 'background')
	OUTPUT_DIR = output_folder

	# 1. Initialize the generator
	lineOCR = HandwrittenLineGenerator(allowed_chars=ALLOWED_CHARS)

	# 2. Load the pkl and text files
	lineOCR.load_character_database(TEST_PKL_FILE)
	lineOCR.load_text_database(TEST_TXT_FILE)
	lineOCR.load_background_image_files(BACKGROUND_NOISES)
	lineOCR.initialize()

	# 3a. Generate images and save to external folder
	lineOCR.generate_images(start=0, label_json=True, save_dir=OUTPUT_DIR)

if __name__ == '__main__':
	## STEP 1: GET AND SAVE CHARACTER IMAGES FROM LINE IMAGES (then check by hand or using models)
	# save images into character folders
	path = 'data/20_Numbers/DIGITAL/data_digital/labeled_folder'
	images, labels = read_images(path)
	for i in range(10):
	    os.mkdir('data/characters/' + str(i))

	save_characters(images, labels)
	# After this step, we need to check character data to make sure that all images are correct.

	# STEP 2: PREPROCESSING & GENERATING LINE IMAGES FROM CHARACTER IMAGESs
	generate()









