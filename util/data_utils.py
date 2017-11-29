import PIL
import numpy as np


def bb_intersection_over_union(box_A, box_B):
    # determine the (x, y)-coordinates of the intersection rectangle
    # Assume box in format [xmin, ymin, xmax, ymax]

    xmin_A = box_A[0]
    ymin_A = box_A[1]
    xmax_A = box_A[2]
    ymax_A = box_A[3]

    xmin_B = box_B[0]
    ymin_B = box_B[1]
    xmax_B = box_B[2]
    ymax_B = box_B[3]

    x1 = max(xmin_A, xmin_B)
    x2 = min(xmax_A, xmax_B)
    y1 = max(ymin_A, ymin_B)
    y2 = min(ymax_A, ymax_B)

    # check if there is no overlap
    if (xmax_A < xmin_B or xmax_A > xmax_B) and \
       (xmax_B < xmin_A or xmax_B > xmax_A) \
            or \
       (ymax_A < ymin_B or ymax_A > ymax_B) and \
       (ymax_B < ymin_A or ymax_B > ymax_A):
        return 0
 
    # compute the area of intersection rectangle
    interArea = (x2 - x1) * (y2 - y1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_AArea = (xmax_A - xmin_A) * (ymax_A - ymin_A)
    box_BArea = (xmax_B - xmin_B) * (ymax_B - ymin_B)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box_AArea + box_BArea - interArea)
 
    # return the intersection over union value
    return iou

def get_pr_curve(gt_boxes, p_boxes, p_scores, p_classes, prs_by_threshold):
    # Assumes boxes, scores and classes are sorted by scores

    precisions = []
    recalls = []
    #thresholds = np.arange(0.0, 1.0, 0.01)
    thresholds = np.arange(0.0, 1.0, 0.1)

    for score_threshold in thresholds:

        tps = 0     # True positives
        fns = 0     # False negatives
        fps = 0     # False postives

        iou_threshold = 0.5

        # Filter for boxes where score is above score_threshold
        boxes_and_classes = np.array(zip(p_boxes, p_classes))
        score_filter = np.array([True if score >= score_threshold else False for score in p_scores])

        # Handle edge cases when no bounding boxes are discovered
        if True not in score_filter:
            precision =  1
            if len(gt_boxes) == 0:
                recall = 1
            else:
                recall =  0
        else:
            filtered_boxes_and_classes = boxes_and_classes[score_filter]
            boxes_available = [1] * len(filtered_boxes_and_classes)
            # Greedily assign predicted boxes to gt_boxes
            for gt_box in gt_boxes:
                detected = False
                # Predicted box must have high IOU with GT and be available
                for i, (p_box, p_class) in enumerate(filtered_boxes_and_classes):
                    iou = bb_intersection_over_union(gt_box[1:], p_box)
                    if iou >= iou_threshold and p_class == gt_box[0] and boxes_available[i]:
                        tps += 1
                        boxes_available[i] = 0
                        detected = True
                        break
                if not detected:
                    fns += 1
            fps += sum(boxes_available)

            # Edge case: If no boxes to recall, recall is 1
            if len(gt_boxes) == 0:
                recall = 1
            else:
                recall = tps / float(tps + fns)

            # Edge case: If no true positives, precision is 1
            if tps == 0:
                precision = 1
            else:
                precision = tps / float(tps + fps)

            assert tps + fps == len(filtered_boxes_and_classes)
            assert tps + fns == len(gt_boxes)

        #print("Threshold: {}, Num predictions: {}, Matches: {}").format(score_threshold,
        #                                                                len(filtered_boxes_and_classes),
        #                                                                tps)

        #print "Precision: {}, Recall: {}, TPs: {}".format(precision, recall, tps)
        '''
        r = round(recall, 1)
        if r not in prs_by_threshold.keys():
            prs_by_threshold[r] = []
        prs_by_threshold[r].append(precision)
        '''
        if score_threshold not in prs_by_threshold.keys():
            prs_by_threshold[score_threshold] = {"p":[], "r":[]}
        prs_by_threshold[score_threshold]["p"].append(precision)
        prs_by_threshold[score_threshold]["r"].append(recall)

    return prs_by_threshold

def get_mAP(prs_by_threshold):
    precisions = []
    recalls = []
    precisions_by_recall = {}

    # Calculate threshold-specific precision and recall values ACROSS frames
    for threshold, prs in prs_by_threshold.iteritems():
        ps = prs["p"]
        rs = prs["r"]
        avg_recall = round(sum(rs) / float(len(rs)), 2)
        avg_precision = sum(ps) / float(len(ps))
        if avg_recall not in precisions_by_recall:
            precisions_by_recall[avg_recall] = []
        precisions_by_recall[avg_recall].append(avg_precision)

    for r, ps in  precisions_by_recall.iteritems():
        #average_precisions = sum(ps) / float(len(ps))
        sorted_ps = sorted(ps, reverse=True)
        myp = 1
        for p in sorted_ps:
            if p == 1:
                continue
            else:
                myp = p
                break
        precisions.append(myp)
        recalls.append(r)
    '''
    for r, ps in  prs_by_threshold.iteritems():
        average_precisions = sum(ps) / float(len(ps))
        precisions.append(average_precisions)
        recalls.append(r)
        #print "Precision: {}, Recall: {}".format(max(ps), r)
    '''
    #mAP = sum(precisions) / float(len(precisions))
    recalls.append(0)
    precisions.append(max(precisions))
    sorted_recalls = sorted(recalls)
    sorted_precisions = [p for _,p in sorted(zip(recalls, precisions))]
    mAP = np.trapz(sorted_precisions, sorted_recalls)
    return mAP, precisions, recalls

def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)
