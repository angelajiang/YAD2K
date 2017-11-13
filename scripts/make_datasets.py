import sys
sys.path.append("util/voc_conversion_scripts")
import os
import voc_to_npz as vnpz

if __name__ == "__main__":
    data_path_base = '/datasets/BigLearning/ahjiang/bb/'

    '''
    dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
    annotations_path = os.path.join(data_path_base,
                                    dataset_path_base,
                                    'annotations/training')
    images_path = os.path.join(data_path_base,
                               dataset_path_base,
                               'images/training')
    labels_path = os.path.join(data_path_base,
                               'udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt')
    scale = 0.5
    dest_file = 'npz/object-detect-crowd-ai-tmp'
    '''

    dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
    annotations_path = os.path.join(data_path_base,
                                    dataset_path_base,
                                    'annotations/training')
    images_path = os.path.join(data_path_base,
                               dataset_path_base,
                               'images/training')
    labels_path = os.path.join(data_path_base,
                               'udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt')
    scale = 0.5
    split = 0.01
    dest_file = 'npz/object-detect-crowd-ai-tmp-' + str(split)

    dest_path = os.path.join(data_path_base, dest_file)
    vnpz.create_npz(images_path,
                    annotations_path,
                    labels_path,
                    dest_path,
                    split_ratio = split,
                    scale = scale,
                    debug = False,
                    shuffle_frames = False)


