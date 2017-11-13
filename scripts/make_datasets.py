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
    split = 0.1
    dest_file = 'npz/object-detect-crowd-ai-tmp'
    '''

    dataset_path_base = 'youtube-bb/youtubebbdevkit2017/youtubebb2017/'
    annotations_path = os.path.join(data_path_base,
                                    dataset_path_base,
                                    'Annotations')
    images_path = os.path.join(data_path_base,
                               dataset_path_base,
                               'JPEGImages')
    labels_path = os.path.join(data_path_base,
                               dataset_path_base,
                               'labels.txt')
    scale = 1
    split = 0.001
    dest_file = 'npz/youtube-bb-car-truck' + str(split)

    dest_path = os.path.join(data_path_base, dest_file)
    vnpz.create_npz(images_path,
                    annotations_path,
                    labels_path,
                    dest_path,
                    target_width = 640,
                    target_height = 400,
                    split_ratio = split,
                    debug = False,
                    shuffle_frames = False)


