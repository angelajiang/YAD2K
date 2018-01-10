import numpy as np
from itertools import compress


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


def get_pr_curve(input_boxes, output_boxes, output_scores, output_classes, iou_threshold=0.5):

    prs_by_threshold = {}
    thresholds = np.arange(0.0, 1.0, 0.01)

    # Hack for binary classifiers where we're only looking for one class
    #for i in range(len(input_boxes)):
    #    if len(input_boxes[i]) > 0 and len(input_boxes[i][0]) > 0:
    #        target_class = input_boxes[i][0][0]
    #        break

    for score_threshold in thresholds:
        
        tps = 0     # True positives
        fns = 0     # False negatives
        fps = 0     # False postives

        for gt_boxes, p_boxes, p_scores, p_classes in \
                zip(input_boxes, output_boxes, output_scores, output_classes):

            # Filter for boxes where score is above score_threshold
            boxes_and_classes = list(zip(p_boxes, p_classes))
            score_filter = np.array([True if score >= score_threshold \
                                          else False \
                                          for score in p_scores])
            #index_filter = np.array([True if c == target_class and score >= score_threshold\
            #                              else False \
            #                              for c, score in zip(p_classes, p_scores)])
            filtered_boxes_and_classes = list(compress(boxes_and_classes, score_filter))

            # Edge case: if there are no boxes, increment nothing
            if len(gt_boxes) == 0:
                fps += len(filtered_boxes_and_classes)
            # Edge case: if no boxes are discovered, increment false negatives
            elif len(filtered_boxes_and_classes) == 0:
                fns += len(gt_boxes)
            else:
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

        # Edge case: If no ground truth positives, recall is 1
        if (tps + fns) == 0:
            recall = 1
        else:
            recall = tps / float(tps + fns)

        # Edge case: If no true positives, precision is 1
        if (tps + fps) == 0:
            precision = 1
        else:
            precision = tps / float(tps + fps)

        #print("Threshold: %.4g , TPs: %d, FPs: %d, FNs: %d" % (score_threshold,
        #                                                       tps,
        #                                                       fps,
        #                                                       fns))


        prs_by_threshold[score_threshold] = {}
        prs_by_threshold[score_threshold]["p"] = precision
        prs_by_threshold[score_threshold]["r"] = recall

    return prs_by_threshold

def get_mAP(prs_by_threshold):
    precisions = []
    recalls = []
    precisions_by_recall = {}

    # Calculate threshold-specific precision and recall values ACROSS frames
    for threshold, prs in prs_by_threshold.items():
        precision = prs["p"]
        recall = prs["r"]
        recall_rounded = round(recall, 2)
        if recall_rounded not in precisions_by_recall:
            precisions_by_recall[recall_rounded] = []
        precisions_by_recall[recall_rounded].append(precision)

    for r, ps in  precisions_by_recall.items():
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

    recalls.append(0)
    precisions.append(max(precisions))
    sorted_recalls = sorted(recalls)
    sorted_precisions = [p for _,p in sorted(zip(recalls, precisions))]
    mAP = np.trapz(sorted_precisions, sorted_recalls)

    return mAP, precisions, recalls

