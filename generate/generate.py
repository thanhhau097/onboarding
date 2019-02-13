from random import randint

with open('dev/text_to_gen.txt', 'w') as f:
    for i in range(1, 11):
        for _ in range(3000):
            f.write(str(randint(10**i, 10**(i+1))))
            f.write('\n')

import glob
import os
import pickle
from dataloader.generate.image import HandwrittenLineGenerator, PrintedLineGenerator
import json 
import cv2
import os 
from tqdm import tqdm

TEST_PKL_FILE='dev/latin_number.pkl'
TEST_TXT_FILE='dev/text_to_gen.txt'
ALLOWED_CHARS='dev/charset_codes.txt'
BACKGROUND_NOISES = 'dev/background'
OUTPUT_DIR='output'

# 1. Initialize the generator
lineOCR = HandwrittenLineGenerator(allowed_chars=ALLOWED_CHARS)


# 2. Load the pkl and text files
lineOCR.load_character_database(TEST_PKL_FILE)
lineOCR.load_text_database(TEST_TXT_FILE)
lineOCR.load_background_image_files(BACKGROUND_NOISES)
lineOCR.initialize()


# 3a. Generate images and save to external folder
lineOCR.generate_images(start=0, label_json=True, save_dir=OUTPUT_DIR)

## DUMP IMAGES
with open('output/labels.json', 'r') as f:
    data = json.load(f)

images = []
labels = []

for key in tqdm(data):
    image = cv2.imread(os.path.join('output', key), 0)
    image = 255 - image
    images.append(image)
    labels.append(data[key])

with open('data/images.pkl', 'wb') as f:
    pickle.dump(images, f)
with open('data/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

    