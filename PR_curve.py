import json
import matplotlib.pyplot as plt
import numpy as np
import copy


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # compare rows first, find which box is below the other
    if box_1[0] >= box_2[2] or box_2[0] >= box_1[2]:
        iou = 0
        return iou

    # compare cols to see if one box to the left or right or other box
    if box_1[1] >= box_2[3] or box_2[1] >= box_1[3]:
        iou = 0
        return iou

    # calculate intersection
    
    intersection_row = abs(max(box_1[0], box_2[0]) - min(box_1[2], box_2[2]))
    intersection_col = abs(max(box_1[1], box_2[1]) - min(box_1[3], box_2[3]))
    
    
    intersection = intersection_row * intersection_col

    # when calculating union dont include intersection area twice
    union = ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1])) + ((box_2[2] - box_2[0]) * (box_2[3] - box_2[1])) - intersection
    iou = intersection/union

    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    local_preds = copy.deepcopy(preds)
    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in local_preds.items():
        gt_new = gts[pred_file]
        
        gt = []
        for i in range(len(gt_new)):
        	gt.append(gt_new[i]['bbox'])



        for i in range(len(gt)):
            # convert the gt to integers (some of them are floats)
            gt[i] = [int(item) for item in gt[i]]

            TP_flag = 0
            for j in range(len(pred)):
                
                if pred[j] == -1:
                    continue

                # take only those predictions which are above the confidence threshold
                if pred[j][-1] < conf_thr:
                	pred[j] = -1
                	continue
                
                iou = compute_iou(pred[j][:4], gt[i])
                
                if iou > iou_thr:

                    # if we already associated a prediction with this gt then do not recount it as TP..instead it will be counted as a FP
                    if TP_flag == 1:
                        continue

                    # if no prediction has been associated with this gt, then count it as TP and remove it from list (so its not counted as a FP for a different gt)
                    TP += 1
                    TP_flag = 1
                    # remove it from our list of preds (set it to -1 for now)
                    pred[j] = -1
            
            # if none of the predictions had an overlap with this particular gt, then we know it is a miss
            if TP_flag == 0:
                FN += 1

        # remove the -1s and whatever remains now in pred are all false positives
        pred_clean = [y for y in pred if y!= -1]
        FP += len(pred_clean)
        #import ipdb;ipdb.set_trace()

    #print (TP, FP, FN, conf_thr)

    '''
    END YOUR CODE
    '''
    return TP, FP, FN





with open('/home/tarun/caltech-ee148/project/output_files/preds_test_manually_annotated.json') as f:
	preds_val = json.load(f)
	
with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_test.json') as f:
	gts_val = json.load(f)



confidence_thrs = []
for fname in preds_val:
    bboxs = preds_val[fname]
    for box in bboxs:
        confidence_thrs.append(box[4])

#confidence_thrs = np.sort(np.array(confidence_thrs))
confidence_thrs = np.linspace( np.max(confidence_thrs),0, num=50)



plt.xlim([0,1])
#plt.ylim([0,1])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision recall curve - ant test set')

for iou_thresh in np.array([0.5]):
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))

    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_val, gts_val, iou_thr=iou_thresh, conf_thr=conf_thr)
        

    # Plot training set PR curves
    precision_list = []
    recall_list = []

    for i in range(0,confidence_thrs.shape[0]):
        print ('###' + str(confidence_thrs[i]) + '###')
        
        precision = tp_train[i]/(tp_train[i] + fp_train[i])
        recall = tp_train[i]/(tp_train[i] + fn_train[i])
        if precision==0 and recall==0:
            continue
        print (precision, recall)

        precision_list.append(precision)
        recall_list.append(recall)

    #print (iou_thresh)
    #print (recall_list, precision_list)
    #import ipdb;ipdb.set_trace()
    plt.plot(np.array(recall_list), np.array(precision_list),c='tab:blue', label=' model trained on courtyard + ant nest data')

plt.legend()
plt.show()


