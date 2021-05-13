import cv2
import numpy as np
from skimage.feature import blob_doh
from skimage import measure
import matplotlib.pyplot as plt
import os
import json
from os import path
from skimage import measure
import matplotlib.pyplot as plt

import skimage
import skimage.morphology

# 2021-05-07_10_31_01
# 2021-05-07_13_00_58
# 2021-05-07_19_01_01


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


## background subtraction 
## subtract the future frames from the current one and use that as a channel
vid_folder = '2021-05-07_19_01_01'
parent_folder = '/home/tarun/caltech-ee148/project/'
cap = cv2.VideoCapture("/home/tarun/Downloads/"+vid_folder+"/vid.h264")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('/home/tarun/Downloads/04_07_2021_13_30_23_cropped_HEQ-MOG-ED_vs_HEQ-MOG-noED.mp4', fourcc, 30, (640*2,480))
frame_number= 0
fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG()

list_of_areas = []

#os.makedirs(parent_folder+vid_folder, exist_ok=True)
# write the annotations
annotations = {}
thresh = 90

while(1):
	if frame_number > 3500:
		break
	ret, frame = cap.read()
	frame = frame[:, 50:-50, :]
	frame  = cv2.resize(frame, (640,480))

	boxes_color = []
	boxes_bg = []

	# histogram EQ - seems like this helps in dark cases but makes things slightly worse when there is lighting changes
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
	frame2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	

	# Background subtraction - MOG2 is more sensitive to light changes. MOG works best 
	fgmask_bgsub = fgbg2.apply(frame2)

	# apply  dilation + erosion (closing) on mask
	kernel = np.ones((2,2), np.uint8)

	img_dilation = cv2.dilate(fgmask_bgsub, kernel, iterations=5)
	img_erosion = cv2.erode(img_dilation, kernel, iterations=5)

	# remove some salt and pepper noise created from above step
	img_erosion = cv2.erode(img_erosion, kernel, iterations=1)
	img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

	fgmask_bgsub = img_dilation

	### color thresholding ####
	b1 = cv2.subtract(frame2[:, :, 2], frame2[:, :, 0])>thresh
	b2 = cv2.subtract(frame2[:, :, 2], frame2[:, :, 1])>thresh

	fgmask_color = 255*(b1&b2).astype(np.uint8)

	img_dilation  = cv2.dilate(fgmask_color, skimage.morphology.selem.disk(5), iterations=1)
	img_erosion = cv2.erode(img_dilation, skimage.morphology.selem.disk(7), iterations=1)

	fgmask_color = img_erosion

	# find the centroids of the blobs created
	contours,hierarchy = cv2.findContours(fgmask_color, 1, 2)

	##### color based approach ########
	idx = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#if area > 10 and area < 1100:
		x,y,w,h = cv2.boundingRect(cnt)
		# center the boxes
		#frame2 = cv2.rectangle(frame2,(x-30,y-30),(x+w+30,y+h+30),(255,0,0),3)
		
		boxes_color.append([x-15,y-15,x+w+15,y+h+15]) 
		#annotations[vid_folder +'_' +str(frame_number)+'.png'].append({'key':idx,'bbox':[x-15,y-15,x+w+15,y+h+15]})
		idx+=1
	######################################

	#### background subtraction approach #######

	contours,hierarchy = cv2.findContours(fgmask_bgsub, 1, 2)
	idx = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 400 and area < 800:
			x,y,w,h = cv2.boundingRect(cnt)
			#frame2 = cv2.rectangle(frame2,(x-5,y-5),(x+w,y+h),(0,255,0),2)
			#annotations[vid_folder +'_' +str(frame_number)+'.png'].append({'key':idx,'bbox':[x,y,x+w,y+h]})
			boxes_bg.append([x-5,y-5,x+w,y+h])
			idx += 1

	
	

	##### now calculate overlap between boxes detected by both approachs and reject double counting (if two boxes overlap then take smaller box)
	if len(boxes_bg) == 0 or len(boxes_color) == 0:
		frame_number += 1
		continue

	annotations[parent_folder + 'images/' + vid_folder + '_' +str(frame_number)+'.png'] = []

	boxes_color = np.array(boxes_color)
	boxes_bg = np.array(boxes_bg)
	idxs_to_reject = []

	for i in range(0,boxes_bg.shape[0]):
		# I am going to use top left corner of each box as an anchor point and find boxes from boxes_color close to that point
		near_x = np.where(abs(boxes_color[:,0] - boxes_bg[i][0]) < 100)[0]
		near_y = np.where(abs(boxes_color[:,1] - boxes_bg[i][1]) < 100)[0]
		box_idxs = np.intersect1d(near_x, near_y)

		# for each of these box_idxs, calculate the overlap between the box_idx and the box_bg...keep the smaller one
		for k in box_idxs:
			iou = compute_iou(list(boxes_bg[i]), list(boxes_color[k]))
			# if there is sufficient overlap, then this means that the bigger box (almost always bg box) should be removed
			if iou >= 0.3:
				idxs_to_reject.append(i)
				continue

	
	## now draw all boxes in boxes_color and all boxes except idxs_to_reject from boxes_bg
	idx = 0
	for b in boxes_color:
		#frame2 = cv2.rectangle(frame2,(b[0],b[1]),(b[2],b[3]),(255,0,0),3)
		annotations[parent_folder + 'images/' + vid_folder + '_' +str(frame_number)+'.png'].append({'key':idx,'bbox':[int(b[0]),int(b[1]),int(b[2]),int(b[3])]})
		idx += 1

	for k,b in enumerate(boxes_bg):
		if k in idxs_to_reject:
			continue
		#frame2 = cv2.rectangle(frame2,(b[0],b[1]),(b[2],b[3]),(0,255,0),3)
		annotations[parent_folder + 'images/' + vid_folder + '_' +str(frame_number)+'.png'].append({'key':idx,'bbox':[int(b[0]),int(b[1]),int(b[2]),int(b[3])]})
		idx += 1


	#cv2.imshow('w', frame2)
	#cv2.waitKey(30)
	#cv2.imwrite(parent_folder + vid_folder + '/' + vid_folder + '_' +str(frame_number)+'.png', frame2)
	cv2.imwrite(parent_folder + 'images/' + vid_folder + '_' +str(frame_number)+'.png', frame2)
	frame_number += 1



# write all annotations (from multiple videos) to json
if path.exists(parent_folder + 'annotations/all.json'):
	with open(parent_folder + 'annotations/all.json', 'r+') as outfile:
		data = json.load(outfile)
		data.update(annotations)
		outfile.seek(0)
		json.dump(data, outfile)
else:
	with open(parent_folder + 'annotations/all.json', 'w') as outfile:
		json.dump(annotations, outfile)

cap.release()
cv2.destroyAllWindows()

	