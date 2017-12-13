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

    training_example = True
    test_example = True

    if training_example:

        ################ Training ################

        split = 0.8

        # atrum - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/atrium'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'atrium_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'atrium_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)
        dest_file = 'npz/urban-tracker-atrium-pedestrian-training'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = True,
                        shuffle_frames = False)

        # Sherbrooke - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/sherbrooke'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'sherbrooke_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'sherbrooke_masked_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)
        dest_file = 'npz/urban-tracker-sherbrooke-pedestrian-training'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_1 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = True,
                        shuffle_frames = False)

        # Rouen - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/rouen'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'rouen_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'rouen_masked_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)
        dest_file = 'npz/urban-tracker-rouen-pedestrian-training'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_2 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = True,
                        shuffle_frames = False)

        # St Marc - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/stmarc'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'stmarc_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'stmarc_masked_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)

        dest_file = 'npz/urban-tracker-stmarc-pedestrian-training'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_3 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = False,
                        shuffle_frames = False)

        # Merge
        dest_file = 'npz/urban-tracker-rouen-sherbrooke-stmarc-pedestrian-training'
        dest_path = os.path.join(data_path_base, dest_file)
        npz_list = [npz_path_1 + ".npz", npz_path_2 + ".npz", npz_path_3 + ".npz"]
        mnpz.merge(npz_list, dest_path)

    if test_example:

        ################ Test ################

        split = 0.2

        # Atrium - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/atrium'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'atrium_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'atrium_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)
        dest_file = 'npz/urban-tracker-atrium-pedestrian-test'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = False,
                        shuffle_frames = False)

        # Sherbrooke - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/sherbrooke'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'sherbrooke_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'sherbrooke_masked_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)
        dest_file = 'npz/urban-tracker-sherbrooke-pedestrian-test'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_1 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = False,
                        shuffle_frames = False)

        # Rouen - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/rouen'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'rouen_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'rouen_masked_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)
        dest_file = 'npz/urban-tracker-rouen-pedestrian-test'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_2 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = False,
                        shuffle_frames = False)

        # St Marc - pedestrian
        labels_set = set()
        labels_set.add("pedestrian")
        labels_file = "pedestrian-label.txt"
        dataset_path_base = 'urban-tracker/stmarc'
        annotations_path = os.path.join(data_path_base,
                                        dataset_path_base,
                                        'stmarc_annotations/annotations')
        images_path = os.path.join(data_path_base,
                                   dataset_path_base,
                                   'stmarc_masked_frames')
        labels_path = os.path.join(data_path_base,
                                   'urban-tracker/labels',
                                   labels_file)

        dest_file = 'npz/urban-tracker-stmarc-pedestrian-test'
        dest_path = os.path.join(data_path_base, dest_file)

        npz_path_3 = vnpz.create_npz(images_path,
                        annotations_path,
                        labels_path,
                        dest_path,
                        labels_set = labels_set,
                        target_width = target_width,
                        target_height = target_height,
                        split_ratio = split,
                        debug = False,
                        start_at_beginning = False,
                        shuffle_frames = False)

        # Merge
        dest_file = 'npz/urban-tracker-rouen-sherbrooke-stmarc-pedestrian-test'
        dest_path = os.path.join(data_path_base, dest_file)
        npz_list = [npz_path_1 + ".npz", npz_path_2 + ".npz", npz_path_3 + ".npz"]
        mnpz.merge(npz_list, dest_path)

