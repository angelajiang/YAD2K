"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from itertools import tee
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model, save_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

sys.path.append("util")

import data_utils

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default=os.path.join('..', 'DATA', 'underwater_data.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))

argparser.add_argument(
    '-p',
    '--model_prefix',
    help='File prefix to save model',
    default='data/yolov2-model-tmp')

argparser.add_argument(
    '-n',
    '--num_frozen',
    help='Number of layers held frozen during training',
    default=0)

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    model_prefix = os.path.expanduser(args.model_prefix)
    num_frozen = int(os.path.expanduser(args.num_frozen))

    model_prefix += "-" + str(num_frozen) + "fr"
    print "Training model:", model_prefix

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    # Load data one checkpoint at a time

    print "Loading", data_path
    data = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'
    print "Data loaded."

    image_data_gen, boxes = data_utils.process_data(iter(data['images']),
                                                data['images'].shape[2],
                                                data['images'].shape[1],
                                                data['boxes'],
                                                dim=608)

    anchors = YOLO_ANCHORS

    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    train(
        class_names,
        anchors,
        image_data_gen,
        boxes,
        detectors_mask,
        matching_true_boxes,
        model_prefix,
        num_frozen
    )

    model_body, model = create_model(anchors, class_names, num_frozen=num_frozen)

    draw(model_body,
        class_names,
        anchors,
        image_data,
        image_set='val', # assumes training/validation split is 0.9
        weights_name='data/checkpoint_best_weights.h5',
        save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [608, 608])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, num_frozen=0):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    num_frozen: number of layers whose weights are held frozen during training

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (19, 19, 5, 1)
    matching_boxes_shape = (19, 19, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(608, 608, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('data', 'model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('data', 'model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    num_layers = len(topless_yolo.layers)
    if num_frozen > num_layers:
        print "Cannot freeze", num_frozen, "/", num_layers, "layers"
        sys.exit()

    print "Freezing", num_frozen, "/", num_layers, "layers"
    for i, layer in enumerate(topless_yolo.layers):
        if i < num_frozen:
            layer.trainable = False

    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def create_generator(images, boxes, detectors_mask, matching_true_boxes, \
                     batch_size=8, start=0, stop=None):

    if stop == None:
        stop = len(boxes)

    img_batch = []
    a_batch = []
    b_batch = []
    c_batch = []
    y_batch = []

    while(True):
        images, images_copy = tee(images)
        for i, (a,b,c) in enumerate(zip(boxes, detectors_mask, matching_true_boxes)):

            if i >= start and i < stop:
                img = next(images_copy)
                img_batch.append(img)
                a_batch.append(a)
                b_batch.append(b)
                c_batch.append(c)
                y_batch.append(0)

                if len(a_batch) == batch_size:
                    tup = ([np.array(img_batch),
                            np.array(a_batch),
                            np.array(b_batch),
                            np.array(c_batch)],
                            np.array(y_batch))
                    yield tup
                    img_batch = []
                    a_batch = []
                    b_batch = []
                    c_batch = []
                    y_batch = []

def train(class_names, anchors, image_data_gen, boxes, detectors_mask,
          matching_true_boxes, model_prefix, num_frozen, validation_split=0.1,
          batch_size=8):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    '''
    logging = TensorBoard()
    checkpoint = ModelCheckpoint("data/checkpoint_best_weights.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model_body, model = create_model(anchors, class_names, load_pretrained=True, num_frozen=num_frozen)

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.
    
    split_index = int(len(boxes) * 0.9)

    num_training = (split_index - 1) / batch_size
    num_validation = (len(boxes) - split_index) / batch_size

    print split_index, num_training, num_validation, len(boxes)

    gen_train = create_generator(image_data_gen, boxes, detectors_mask, matching_true_boxes, stop = split_index, batch_size=batch_size)
    gen_test = create_generator(image_data_gen, boxes, detectors_mask, matching_true_boxes, start = split_index, batch_size=batch_size)

    model.fit_generator(gen_train,
                        epochs=40,
                        validation_data = gen_test,
                        steps_per_epoch = num_training,
                        validation_steps = num_validation,
                        callbacks=[logging, checkpoint, early_stopping])

    #model.fit([list(image_data_gen), boxes, detectors_mask, matching_true_boxes],
    #          np.zeros(len(image_data)),
    #          validation_split=0.1,
    #          batch_size=8,
    #          epochs=40,
    #          callbacks=[logging, checkpoint, early_stopping])

    #sess = K.get_session()
    #graph_def = sess.graph.as_graph_def()
    #tf.train.write_graph(graph_def,
    #                     logdir='.',
    #                     name=model_prefix+".pb",
    #                     as_text=False)
    #saver = tf.train.Saver()
    #saver.save(sess, model_prefix+'.ckpt', write_meta_graph=True)

    model_body.save_weights(model_prefix+"_weights.h5")
    save_model(model_body, model_prefix+".h5", overwrite=True)

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='data/checkpoint_best_weights.h5', out_path="data/output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
