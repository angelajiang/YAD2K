#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import cv2
import imghdr
import os
import random

import numpy as np
import keras
from keras import backend as K
from keras.models import load_model, save_model
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Lambda

import sys
sys.path.append("util")
import data_utils

from yad2k.models.keras_yolo import yolo_head, yolo_eval, yolo_post_process

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    '-m',
    '--model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='data/model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='data/model_data/coco_classes.txt')
parser.add_argument(
    '--mode',
    help='{0: dir, 1:npz}',
    default='0')
parser.add_argument(
    '-t',
    '--test_dir',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='images/out')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)


def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)

    input_mode = int(os.path.expanduser(args.mode))
    assert input_mode == 0 or input_mode == 1, 'Input mode must be in {0,1}'
    test_path = os.path.expanduser(args.test_dir)

    output_path = os.path.expanduser(args.output_path)
    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.

    input_image_shape = K.placeholder(shape=(2, ))

    boxes, scores, classes = yolo_post_process(yolo_model.output,
                                               anchors,
                                               len(class_names),
                                               input_image_shape,
                                               args.score_threshold,
                                               args.iou_threshold)


    total_num_bboxes = 0
    total_num_images = 0

    # "Dir" mode
    if input_mode == 0:
        for image_file in os.listdir(test_path):
            try:
                image_type = imghdr.what(os.path.join(test_path, image_file))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue

            image = Image.open(os.path.join(test_path, image_file))
            if is_fixed_size:  # TODO: When resizing we can use minibatch input.
                resized_image = image.resize(
                    tuple(reversed(model_image_size)), Image.BILINEAR)
                image_data = np.array(resized_image, dtype='float32')
            else:
                # Due to skip connection + max pooling in YOLO_v2, inputs must have
                # width and height as multiples of 32.
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                resized_image = image.resize(new_image_size, Image.BILINEAR)
                image_data = np.array(resized_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

            total_num_bboxes += len(out_boxes)
            total_num_images += 1 

            # Rank boxes, scores and classes by score

            font = ImageFont.truetype(font='font/FiraMono-Medium.otf', \
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box
                print predicted_class

	        label = '{} {:.2f}'.format(predicted_class, score)

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            image.save(os.path.join(output_path, image_file), quality=90)

    # "NPZ" mode
    elif input_mode == 1:
        # 2 modes: 1) "dir" inference from dir 2) "npz" inference and mAP from bounding boxes
        data = np.load(test_path) # custom data saved as a numpy file.
        input_images = data['images']
        input_boxes = data['boxes']
        mAPs = []

        total_num_bboxes = 0

        for image, gt_boxes in zip(input_images, input_boxes):

            height, width = image.shape[:2]

            if is_fixed_size:
                width_scale = model_image_size[1] / float(width) 
                height_scale = model_image_size[0] / float(height) 
                resized_image = cv2.resize(image,
                                        tuple(reversed(model_image_size)),
                                        interpolation = cv2.INTER_LINEAR)
            else:
                # Due to skip connection + max pooling in YOLO_v2, inputs must have
                # width and height as multiples of 32.
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                width_scale = new_image_size[0]
                height_scale = new_image_size[1]
                resized_image = image.resize(new_image_size, Image.BILINEAR)
            
            image_data = np.array(resized_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.shape[0], image.shape[1]],
                    K.learning_phase(): 0
                })

            total_num_bboxes += len(out_boxes)
            total_num_images += 1 

            transformed_out_boxes = []
            for out_box in out_boxes:
                new_box = [0,0,0,0]
                xmin = out_box[1]
                xmax = out_box[3]
                ymin = out_box[0]
                ymax = out_box[2]

                new_box[0] = xmin
                new_box[1] = ymin
                new_box[2] = xmax
                new_box[3] = ymax

                transformed_out_boxes.append(new_box)

            ap = data_utils.calculate_mAP(gt_boxes,
                                           transformed_out_boxes,
                                           out_scores,
                                           out_classes)
            mAPs.append(ap)

        mAP = sum(mAPs) / float(len(mAPs))
        print "mAP: {}".format(mAP)

    print('Found {} boxes in {} images'.format(total_num_bboxes, total_num_images))

    sess.close()

if __name__ == '__main__':
    _main(parser.parse_args())
