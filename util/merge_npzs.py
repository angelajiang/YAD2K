import numpy as np

def merge(npz_list, dest_path):
    first = True
    all_boxes = []

    for npz_file in npz_list:
        data = np.load(npz_file)
        images = data['images']
        boxes = data['boxes']
        for box in boxes:
            all_boxes.append(box)
        if first:
            all_images = images
            first = False
        else:
            all_images = np.concatenate((all_images, images))

    all_boxes = [np.array(i) for i in all_boxes]
    all_boxes = np.array(all_boxes)

    #save dataset
    print("Saving %d images" % len(all_images))
    np.savez(dest_path, images=all_images, boxes=all_boxes)
    print('Data saved: ', dest_path + ".npz")
    return dest_path
