import numpy as np

def merge(npz_list, dest_path):
    first = True
    all_boxes = []
    all_images = []

    for npz_file in npz_list:
        data = np.load(npz_file)
        images = data['images']
        boxes = data['boxes']
        for image, box in zip(images, boxes):
            all_images.append(image)
            all_boxes.append(box)
            #all_images = np.concatenate((all_images, images))

    all_boxes = [np.array(i) for i in all_boxes]
    all_boxes = np.array(all_boxes)

    all_images = [np.array(i) for i in all_images]
    all_images = np.array(all_images)

    print(all_images.shape)
    print(all_boxes.shape)

    #save dataset
    print("Saving %d images" % len(all_images))
    np.savez(dest_path, images=all_images, boxes=all_boxes)
    print('Data saved: ', dest_path + ".npz")
    return dest_path
