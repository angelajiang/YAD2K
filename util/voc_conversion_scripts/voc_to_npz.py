import os
import sys
import json
import glob
import cv2
import PIL.Image
import numpy as np
from random import shuffle

from lxml import etree

def split(split_ratio, annotations_path, shuffle_frames_frames = True):

    assert split_ratio >= 0 and split_ratio <= 1, 'split_ratio should be between 0 and 1'

    # Get image filenames to shuffle_frames
    fnames = []
    for the_file in os.listdir(annotations_path):
        file_path = os.path.join(annotations_path, the_file)
        if os.path.isfile(file_path) and the_file.endswith(".xml"):
            fnames.append(the_file)

    if shuffle_frames_frames:
        shuffle(fnames)

    num_images = int(split_ratio * len(fnames))

    print("Returning %d images" % num_images)

    return fnames[:num_images]

def create_npz(images_path, annotations_path, labels_path, dest_path,
               target_width = 640, target_height = 400,
               split_ratio = 1, debug = False, shuffle_frames = False):
    # Debug only loads 10 images

    text = []
    image_labels = []
    label_indices = {}

    with open(labels_path) as f:
        for i, line in enumerate(f):
            label = line.rstrip()
            label_indices[label] = i

    annotations_to_include = split(split_ratio, annotations_path, shuffle_frames)

    # load images
    images = []

    i = 0
    for fname in os.listdir(annotations_path):
        if fname in annotations_to_include:
            filename = os.path.join(annotations_path, fname)

            # Parse images
            img = cv2.imread(os.path.join(images_path, os.path.basename(filename).split('.')[0] + ".jpg"))

            old_shape = img.shape
            width, height = img.shape[:2]
            width_scale = target_width / float(width) 
            height_scale = target_height / float(height) 

            img = cv2.resize(img, (target_width, target_height))

            try:
                (b, g, r)=cv2.split(img)
                img = cv2.merge([r,g,b])
            except:
                print(i)

            print(old_shape, img.shape)
            images.append(img)

            # Parse annotations
            image_labels.append([])  
            image_labels[i].append([images_path, os.path.basename(filename).split('.')[0] + ".jpg"])

            tree = etree.parse(filename)
            root = tree.getroot()

            for j,object in enumerate(root.findall('object')) :

                boxConfig = []
                # Classe of the object
                class_str = object.find('name').text
                class_index = label_indices[class_str]
                boxConfig.append(class_index)

                box = object.find('bndbox')
                boxConfig.append(float(box.find('xmin').text) * width_scale)
                boxConfig.append(float(box.find('ymin').text) * width_scale)
                boxConfig.append(float(box.find('xmax').text) * height_scale)
                boxConfig.append(float(box.find('ymax').text) * height_scale)

                image_labels[i].append(boxConfig)

            i += 1
            if debug and i == 10:
                break

    #convert to numpy for saving
    images = np.array(images, dtype=np.uint8)
    image_labels = [np.array(i[1:]) for i in image_labels]# remove the file names
    image_labels = np.array(image_labels)

    print("Saving %d images" % len(images))

    #save dataset
    np.savez(dest_path, images=images, boxes=image_labels)
    print('Data saved: ', dest_path + ".npz")
