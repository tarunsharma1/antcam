import cv2
import numpy as np
from skimage.feature import blob_doh
from skimage import measure
import matplotlib.pyplot as plt

import skimage
import skimage.morphology

import tqdm

cap = cv2.VideoCapture("D:/Parker_lab/data/2021-05-07_17_31_01/vid.h264")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('/home/tarun/Downloads/04_07_2021_13_30_23_cropped_HEQ-MOG-ED_vs_HEQ-MOG-noED.mp4', fourcc, 30, (640*2,480))
frame_number= 0

list_of_areas = []

bg_stride = 200
thresh = 40
end = 1000 

for i in tqdm.tqdm(range(end)):

    ret, frame = cap.read()
    frame = frame[:, 50:-50, :]
    #frame = cv2.resize(frame, (640,480))

    # histogram EQ - seems like this helps in dark cases but makes things slightly worse when there is lighting changes
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    frame2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    b1 = cv2.subtract(frame[:, :, 2], frame[:, :, 0])>thresh
    b2 = cv2.subtract(frame[:, :, 2], frame[:, :, 1])>thresh
    
    fgmask2 = 255*(b1&b2).astype(np.uint8)
    
    fgmask2  = cv2.dilate(fgmask2, skimage.morphology.selem.disk(5), iterations=1)
    fgmask2 = cv2.erode(fgmask2, skimage.morphology.selem.disk(7), iterations=1)
    
    # find the centroids of the blobs created
    contours,hierarchy = cv2.findContours(fgmask2, 1, 2)
    
    for cnt in contours:
        
        #M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        # reject small rectangles and large rectangles (more than one ant)
        list_of_areas.append(area)
        if area > 10 and area < 1100:
            x,y,w,h = cv2.boundingRect(cnt)
            frame2 = cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),3)

    
    cv2.imshow('w', frame2)
    cv2.waitKey(30)
    
plt.hist(list_of_areas, bins = np.unique(np.array(list_of_areas)).shape[0])
plt.show()

cap.release()
cv2.destroyAllWindows()