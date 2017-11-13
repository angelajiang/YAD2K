import os
import sys
sys.path.append("util/voc_conversion_scripts")
sys.path.append("util")

import voc_to_npz as vnpz
import merge_npzs as mnpz

if __name__ == "__main__":
    data_path_base = '/datasets/BigLearning/ahjiang/bb/'
    target_width = 640
    target_height = 400

    # CrowdAI
    dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
    annotations_path = os.path.join(data_path_base,
                                    dataset_path_base,
                                    'annotations/training')
    images_path = os.path.join(data_path_base,
                               dataset_path_base,
                               'images/training')
    labels_path = os.path.join(data_path_base,
                               'udacity-od-crowdai/Udacity_object_dataset/crowdai-labels.txt')
    split1 = 0.01
    dest_file = 'npz/object-detect-crowd-ai-tmp'
    dest_path = os.path.join(data_path_base, dest_file)

    npz_path_1 = vnpz.create_npz(images_path,
                    annotations_path,
                    labels_path,
                    dest_path,
                    target_width = target_width,
                    target_height = target_height,
                    split_ratio = split1,
                    debug = False,
                    shuffle_frames = False)


    # Youtube-BB
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
    split2 = 0.01
    dest_file = 'npz/youtube-bb-car-truck-tmp'
    dest_path = os.path.join(data_path_base, dest_file)

    npz_path_2 = vnpz.create_npz(images_path,
                    annotations_path,
                    labels_path,
                    dest_path,
                    target_width = target_width,
                    target_height = target_height,
                    split_ratio = split2,
                    debug = False,
                    shuffle_frames = False)

    # Merge
    dest_file = 'npz/crowdai-' + str(split1) + '-youtubebb-' + str(split2)
    dest_path = os.path.join(data_path_base, dest_file)
    npz_list = [npz_path_1 + ".npz", npz_path_2 + ".npz"]
    mnpz.merge(npz_list, dest_path)

