import os
import sys
sys.path.append("util/voc_conversion_scripts")
sys.path.append("util")

import voc_to_npz as vnpz
import merge_npzs as mnpz

if __name__ == "__main__":
    data_path_base = '/datasets/BigLearning/ahjiang/bb/'
    target_width = 400
    target_height = 640

    merge_example = False
    crowdai_training_example = False
    crowdai_test_example = False
    crowdai_pedestrian_example = False
    crowdai_pedestrian_augment_example = True

    if merge_example:
        # CrowdAI
        dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations/training')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images/training')
        labels_path = os.path.join(data_path_base,
                                   'udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt')
        split1 = 0.6
        dest_file = 'npz/object-detect-crowd-ai-tmp'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_1 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split1,
                        scale = 0.5,
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
        split2 = 0.1
        dest_file = 'npz/youtube-bb-pedestrian-pedestrian-tmp'
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
    
    elif crowdai_training_example:

        # CrowdAI only
        dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations/training')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images/training')
        labels_path = os.path.join(data_path_base,
                                   'udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt')
        split = 0.8
        dest_file = 'npz/crowdai-0.8'
        dest_path = os.path.join(data_path_base, dest_file)

        vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        scale = 0.5,
                        debug = False,
                        shuffle_frames = False)

    elif crowdai_test_example:

        # CrowdAI test
        dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations/test')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images/test')
        labels_path = os.path.join(data_path_base,
                                   'udacity-od-crowdai/Udacity_object_dataset/labels/crowdai-labels.txt')
        split = 1
        dest_file = 'npz/crowdai-test'
        dest_path = os.path.join(data_path_base, dest_file)

        vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        scale = 0.5,
                        debug = False,
                        shuffle_frames = False)

    elif crowdai_pedestrian_example:

        # Training
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"

        dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
        labels_path = os.path.join(data_path_base,
                                   'udacity-od-crowdai/Udacity_object_dataset/labels/',
                                   labels_file)

        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations/training')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images/training')
        split = 0.4
        dest_file = 'npz/crowdai-pedestrian-training-' + str(split)

        dest_path = os.path.join(data_path_base, dest_file)

        vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        labels_set = labels_set,
                        scale = 0.5,
                        debug = False,
                        shuffle_frames = False)

        # Test
        dest_file = 'npz/crowdai-pedestrian-test'
        dest_path = os.path.join(data_path_base, dest_file)

        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations/test')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images/test')
        split = 1

        #vnpz.create_npz(images_path,
        #                annotations_path,
        #                labels_path,
        #                dest_path,
        #                target_width = target_width,
        #                target_height = target_height,
        #                split_ratio = split,
        #                labels_set = labels_set,
        #                scale = 0.5,
        #                debug = False,
        #                shuffle_frames = False)


    elif crowdai_pedestrian_augment_example:

        # CrowdAI pedestrian dataset
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"

        dataset_path_base = 'udacity-od-crowdai/object-detection-crowdai-scaled/'
        labels_path = os.path.join(data_path_base,
                                   'udacity-od-crowdai/Udacity_object_dataset/labels/',
                                   labels_file)

        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations/training')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images/training')
        split1 = 0.1
        dest_file = 'npz/crowdai-pedestrian-training-' +str(split1)
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_1 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split1,
                        labels_set = labels_set,
                        scale = 0.5,
                        debug = False,
                        shuffle_frames = False)

        # Caltech pedestrian dataset
        dataset_path_base = 'caltech-ped-data'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'images')
        split2 = 1
        dest_file = 'npz/caltech-pedestrian-training-'+str(split2)
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
        dest_file = 'npz/pedestrian-training-crowdai-'+str(split1)+"-caltech-"+str(split2)
        dest_path = os.path.join(data_path_base, dest_file)
        npz_list = [npz_path_1 + ".npz", npz_path_2 + ".npz"]
        mnpz.merge(npz_list, dest_path)
    
