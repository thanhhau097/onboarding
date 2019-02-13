folder = 'data/20_Numbers/E2ENumber'

import os
import subprocess
from tqdm import tqdm 
import pandas as pd
import cv2


def extracted_data_from_xlsx_file(folder):
    for sub_folder in (os.listdir(folder)):
        os.mkdir(os.path.join(folder, sub_folder, 'extracted'))
        for file_name in tqdm(os.listdir(os.path.join(folder, sub_folder))):
            if 'xlsx' in file_name:
                xlsx_file_path = os.path.join(folder, sub_folder, file_name)
                extracted_file_path = os.path.join(folder, sub_folder, 'extracted', file_name.split('.xlsx')[0])

                os.mkdir(extracted_file_path)
                # extracted 
                subprocess.run(["unzip", xlsx_file_path, "-d", extracted_file_path])

def get_labels_from_file(xlsx_path):
    xl_file = pd.ExcelFile(xlsx_path)

    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    sheet = dfs['Sheet1']
    
    column_image_name = sheet.columns.values[0]
    column_name = sheet.columns.values[-1]
    
    image_names = sheet[sheet.index % 10 == 9][column_image_name].values
    labels = sheet[sheet.index % 10 == 9][column_name].values
    
    
    labels = [str(label).replace(' ', '') for label in labels]
    image_names = [str(image_name) for image_name in image_names]

    filter_names = []
    filter_labels = []
    
    for i in range(len(image_names)):
        name = image_names[i]
        if ('.' in name) and (name != 'nan'):
            filter_names.append(name)
            filter_labels.append(labels[i])
    
    return filter_labels#, filter_names

def get_images_from_extracted_folder(folder):
    images = []
    
    for i in range(len(os.listdir(folder))):
        name = 'image' + str(i+1) + '.png'
        
        image_path = os.path.join(folder, name)
        image = cv2.imread(image_path, 0)
        images.append(image)
        
    return images 

def get_images_and_labels(folder):
    total_false = 0
    
    all_images = []
    all_labels = []
    
    for sub_folder in (os.listdir(folder)):
        for file_name in tqdm(os.listdir(os.path.join(folder, sub_folder))):
            if 'xlsx' in file_name:
                xlsx_file_path = os.path.join(folder, sub_folder, file_name)
                extracted_images_folder = os.path.join(folder, sub_folder, 'extracted', file_name.split('.xlsx')[0], 'xl/media')
                
                labels = get_labels_from_file(xlsx_file_path)
                images = get_images_from_extracted_folder(extracted_images_folder)
                
                if len(labels) != len(os.listdir(extracted_images_folder)):
                    print(xlsx_file_path)
                    print('FALSE\t -----> labels:', len(labels), 'images:', len(os.listdir(extracted_images_folder)))
                    total_false += 1
                    continue
                else:
                    all_images.append(images)
                    all_labels.append(labels)

    return all_images, all_labels



## Cutting images using histogram 
def get_cuting_point_left_border(histogram, threshold):
    met_greater = False
    
    for i, value in enumerate(histogram):
        if not met_greater:
            if value > threshold:
                met_greater = True
        else:
            if value <= threshold:
                point = i
                break
    return point

def vertical_cut(image, threshold=230):
    hist = np.sum(images[index] < threshold, axis=0)
    
    max_value = max(hist)
    min_value = min(hist)
    
    histogram_threshold = (max_value + min_value) * 1/3
    
    # cut into one
    first_point = get_cuting_point_left_border(hist, histogram_threshold)
    second_point = len(hist) - get_cuting_point_left_border(list(reversed(hist)), histogram_threshold)

    return first_point, second_point

def horizontal_cut(image, threshold=230):
    hist = np.sum(images[index] < threshold, axis=1)
    max_value = max(hist)
    min_value = min(hist)
    
    histogram_threshold = (max_value + min_value) * 1/3
    
    # step 1: cut into two image (get first image)
    image1_first_point = 0
    for i, value in enumerate(hist):
        if value > histogram_threshold:
            image1_second_point = i - 1
            break
            
    # step 2: use same vertical cut
    new_hist = hist[image1_second_point:]
    image_2_first_point = image1_second_point + 1 + get_cuting_point_left_border(new_hist, histogram_threshold)
    image_2_second_point = image1_second_point - 1 + len(new_hist) - get_cuting_point_left_border(list(reversed(new_hist)), histogram_threshold)
    
    return image1_first_point, image1_second_point, image_2_first_point, image_2_second_point

def cut_image(image):
    x1, x2, x3, x4 = horizontal_cut(image)
    y1, y2 = vertical_cut(image)
    
    image_1 = image[x1: x2, y1: y2]
    image_2 = image[x3: x4, y1: y2]
    
    return image_1, image_2

def preprocess(images, labels):
    printed_images = []
    hand_writing_images = []
    filter_labels = []
    
    for i, image in tqdm(enumerate(images)):
        try:
            printed_image, hand_writing_image = cut_image(image)
            printed_images.append(printed_image)
            hand_writing_images.append(hand_writing_image)
            filter_labels.append(labels[i])
        except:
            continue
       
    return printed_images, hand_writing_images, filter_labels

if __name__ == '__main__':
	images, labels = get_images_and_labels(folder)

	printed_images, hand_writing_images, filter_labels = preprocess(images, labels)

	with open('data/e2e_processed/printed_images.pkl', 'wb') as f:
	    pickle.dump(printed_images, f)
	with open('data/e2e_processed/hand_writing_images.pkl', 'wb') as f:
	    pickle.dump(hand_writing_images, f)
	with open('data/e2e_processed/labels.pkl', 'wb') as f:
	    pickle.dump(filter_labels, f)








