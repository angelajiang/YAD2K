from random import shuffle
import shutil
import os
import sys

def split(dest_script_file, test_split, training_split,
          annotations_src_path, annotations_dest_path,
          images_src_path, images_dest_path):

    # Get image filenames to shuffle
    fnames = []
    for the_file in os.listdir(images_src_path):
        file_path = os.path.join(images_src_path, the_file)
        if os.path.isfile(file_path) and the_file.endswith(".jpg"):
            fnames.append(the_file.replace(".jpg", ""))

    shuffle(fnames)

    num_test_images = int(test_split * len(fnames))
    num_training_images = int(training_split * len(fnames))

    if num_training_images + num_test_images > len(fnames):
        print "Training and test splits don't make sense"
        sys.exit()

    print num_training_images, "training images"
    print num_test_images, "test images"

    for folder in [annotations_dest_path, images_dest_path]:
    # Remove existing contents in subdirectories training and test
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    os.system("mkdir %s" %file_path)
            except Exception as e:
                print(e)

    with open(dest_script_file, "w+") as f:
        for i, name in enumerate(fnames):
            if i < num_training_images:
                line1 = "cp " + os.path.join(annotations_src_path, name) + ".xml " \
                        + os.path.join(annotations_dest_path, "training") + "\n"
                line2 = "cp " + os.path.join(images_src_path, name) + ".jpg  " \
                        + os.path.join(images_dest_path, "training") + "\n"
            else:
                line1 = "cp " + os.path.join(annotations_src_path, name) + ".xml  " \
                        + os.path.join(annotations_dest_path, "test") + "\n"
                line2 = "cp " + os.path.join(images_src_path, name) + ".jpg  "\
                        + os.path.join(images_dest_path, "test") + "\n"
            f.write(line1)
            f.write(line2)

if __name__ == "__main__":
    split("generate_data_tmp.sh", .2, .3,
          "/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/Udacity_object_dataset/object-detection-crowdai/annotations/",
          "/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-full/annotations/",
          "/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai/",
          "/datasets/BigLearning/ahjiang/bb/udacity-od-crowdai/object-detection-crowdai-full/images/")
