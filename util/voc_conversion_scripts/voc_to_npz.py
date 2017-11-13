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
               split_ratio = 1, scale = 1, debug = False, shuffle_frames = False):
    # Debug only loads 10 images

    if scale != 1:
        print("[WARNING] Scaling annotations by", scale)

    text = []
    image_labels = []
    label_indices = {}

    with open(labels_path) as f:
        for i, line in enumerate(f):
            label = line.rstrip()
            label_indices[label] = i

    annotations_to_include = split(split_ratio, annotations_path, shuffle_frames)

    i = 0
    for fname in os.listdir(annotations_path):
        if fname in annotations_to_include:
            filename = os.path.join(annotations_path, fname)
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
                boxConfig.append(float(box.find('xmin').text) * scale)
                boxConfig.append(float(box.find('ymin').text) * scale)
                boxConfig.append(float(box.find('xmax').text) * scale)
                boxConfig.append(float(box.find('ymax').text) * scale)

                image_labels[i].append(boxConfig)
            i += 1


    # load images
    images = []
    for i, label in enumerate(image_labels):
        #img = np.array(PIL.Image.open(os.path.join(label[0][0], label[0][1])).resize((640, 480)), dtype=np.uint8)
        img = cv2.imread(os.path.join(label[0][0], label[0][1]))
        #img = cv2.resize(img, (640, 480))
        #img = np.array(PIL.Image.open(os.path.join(label[0][0], label[0][1])), dtype=np.uint8)

        try:
            (b, g, r)=cv2.split(img)
            img = cv2.merge([r,g,b])
        except:
            print(i)

        print(img.shape)

        images.append(img)
        if debug and i == 9:
            break

    #convert to numpy for saving
    print(len(images))
    images = np.array(images, dtype=np.uint8)
    image_labels = [np.array(i[1:]) for i in image_labels]# remove the file names
    image_labels = np.array(image_labels)

    print("Saving %d images" % len(images))

    #save dataset
    np.savez(dest_path, images=images, boxes=image_labels)
    print('Data saved: ', dest_path + ".npz")
