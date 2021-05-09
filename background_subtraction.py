import cv2
import numpy as np
from skimage.feature import blob_doh
from skimage import measure
import matplotlib.pyplot as plt

## background subtraction 
## subtract the future frames from the current one and use that as a channel

cap = cv2.VideoCapture("/home/tarun/Downloads/2021-05-07_13_00_58/vid.h264")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('/home/tarun/Downloads/04_07_2021_13_30_23_cropped_HEQ-MOG-ED_vs_HEQ-MOG-noED.mp4', fourcc, 30, (640*2,480))
frame_number= 0
fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG()

list_of_areas = []

while(1):
	if frame_number > 500:
		break
	ret, frame = cap.read()
	frame  = cv2.resize(frame, (640,480))


	# histogram EQ - seems like this helps in dark cases but makes things slightly worse when there is lighting changes
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
	frame2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# Background subtraction - MOG2 is more sensitive to light changes. MOG works best 
	fgmask2 = fgbg2.apply(frame2)

	# apply  dilation + erosion (closing) on mask
	kernel = np.ones((2,2), np.uint8)

	img_dilation = cv2.dilate(fgmask2, kernel, iterations=5)
	img_erosion = cv2.erode(img_dilation, kernel, iterations=5)

	# remove some salt and pepper noise created from above step
	img_erosion = cv2.erode(img_erosion, kernel, iterations=1)
	img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

	fgmask2 = img_dilation
	
	#gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# find the centroids of the blobs created
	contours,hierarchy = cv2.findContours(fgmask2, 1, 2)
	
	for cnt in contours:
		
		#M = cv2.moments(cnt)
		area = cv2.contourArea(cnt)
		# reject small rectangles and large rectangles (more than one ant)
		list_of_areas.append(area)
		if area > 400 and area < 1100:
			x,y,w,h = cv2.boundingRect(cnt)
			frame2 = cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,0,0),2)

	
	cv2.imshow('w', frame2)
	cv2.waitKey(30)


	# new_image2 = np.zeros_like(frame)
	# new_image2[:,:,0] = gray
	# new_image2[:,:,1] = fgmask2
	# new_image2[:,:,2] = gray

	frame_number+=1
	
plt.hist(list_of_areas, bins = np.unique(np.array(list_of_areas)).shape[0])
plt.show()

cap.release()
cv2.destroyAllWindows()

