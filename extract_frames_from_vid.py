import cv2
import numpy as np

#### extract frames from video for annotations ########
vid_folder = '2021-05-13_06_01_00'
parent_folder = '/home/tarun/Downloads/ant-data/beer-tree/'
cap = cv2.VideoCapture(parent_folder+vid_folder+"/vid.h264")

frame_number= 0



#os.makedirs('/home/tarun/caltech-ee148/project/images_beertree_vids/'+vid_folder, exist_ok=True)
# pick out random frame numbers to save
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
vid_out = cv2.VideoWriter(parent_folder + vid_folder + 'cropped.mp4',fourcc, 30.0, (640,480))


while(1):
	ret, frame = cap.read()
	#if frame_number < 3100:
	# 	frame_number+=1
	# 	continue

	if frame_number > 1000:
	 	break
	

	frame  = cv2.resize(frame, (640,480))
	
	vid_out.write(frame)
	
	#cv2.imwrite('/home/tarun/caltech-ee148/project/images_beertree_vids/' + vid_folder + '_' + str(frame_number)+'.png', frame)
	#cv2.imshow('w',frame)
	#cv2.waitKey(30)
	print (frame_number)
	frame_number+=1

vid_out.release()