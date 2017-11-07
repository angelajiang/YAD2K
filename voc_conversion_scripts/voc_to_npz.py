import os
import sys
import json
import glob
import cv2
import PIL.Image
import numpy as np

from lxml import etree

debug = False #only load 10 images
shuffle = False # shuffle dataset

text = []
image_labels = []
dataset_path_base = '/users/ahjiang/image-data/bb/udacity-od-crowdai/object-detection-crowdai-scaled/'
annotations_path = os.path.join(dataset_path_base, 'annotations/training')
images_path = os.path.join(dataset_path_base, 'images/training')
scale = 0.5
if scale != 1:
    print("[WARNING] Scaling annotations by", scale)
print(images_path)

for i,filename in enumerate(glob.glob(annotations_path+'/*.xml')):   
    image_labels.append([])  
    image_labels[i].append([images_path, os.path.basename(filename).split('.')[0] + ".jpg"])
    tree = etree.parse(filename)
    root = tree.getroot()

    for j,object in enumerate(root.findall('object')) :

        boxConfig = []
        # Classe of the object
        boxConfig.append(0)

        box = object.find('bndbox')
        boxConfig.append(float(box.find('xmin').text) * scale)
        boxConfig.append(float(box.find('ymin').text) * scale)
        boxConfig.append(float(box.find('xmax').text) * scale)
        boxConfig.append(float(box.find('ymax').text) * scale)

        image_labels[i].append(boxConfig)


# load images
images = []
image_labels_real = []
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
    #image_labels_real(image_labels[i])
    if debug and i == 9:
        break

#convert to numpy for saving
print(len(images))
images = np.array(images, dtype=np.uint8)
image_labels = [np.array(i[1:]) for i in image_labels]# remove the file names
image_labels = np.array(image_labels)

#shuffle dataset
if shuffle:
    np.random.seed(13)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images, image_labels = images[indices], image_labels[indices]

#save dataset
np.savez("my_dataset", images=images, boxes=image_labels)
print('Data saved: my_dataset.npz')
