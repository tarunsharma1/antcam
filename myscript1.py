import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode


def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0
	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
		# increment the total number of frames read
		total += 1
	# return the total number of frames in the video file
	return total




def get_ant_dicts(img_dir, annotations_dir, split):

    json_file = os.path.join(annotations_dir, split+".json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.keys()):
        record = {}
        
        filename = v
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = imgs_anns[v]
        if annos == []:
        	continue

        objs = []
        for anno in annos:
            
            x1, y1, x2, y2 = anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], anno['bbox'][3] 
            obj = {
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


image_path = '/home/tarun/caltech-ee148/project/images_beertree_vids'
annotations_path = '/home/tarun/caltech-ee148/project/annotations'
training_json = 'manually_annotated_train'
val_json = 'manually_annotated_val'
test_json = 'manually_annotated_test'

for d in [training_json, val_json, test_json]:
    DatasetCatalog.register("ant_" + d, lambda d=d: get_ant_dicts(image_path, annotations_path, d))
    MetadataCatalog.get("ant_" + d).set(thing_classes=[""])
    MetadataCatalog.get("ant_" + d).set(thing_colors=[[0,255,0]])


MetadataCatalog.get("ant_" + val_json).set(evaluator_type = 'coco')
ant_metadata = MetadataCatalog.get("ant_" + training_json)

cfg = get_cfg()
cfg.merge_from_file("/home/tarun/caltech-ee148/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")

cfg.DATASETS.TRAIN = ("ant_" + training_json,)
cfg.DATASETS.TEST = ("ant_" + val_json,)
cfg.TEST.EVAL_PERIOD = 200
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "/home/tarun/Downloads/model_final_721ade.pkl"
### initilize using pretrained courtyard weights 
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_courtyard_3vids.pth")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 3700 # 40 epochs; 119110 was 10 epochs for all courtyard data
#cfg.SOLVER.STEPS = [50000, 100000]        # do not decay learning rate
cfg.SOLVER.STEPS = [500, 2500]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.SOLVER.CHECKPOINT_PERIOD = 500
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


#### to check if annotations are already before starting training #########
# dataset_dicts = get_ant_dicts(image_path, annotations_path, training_json)

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=ant_metadata, scale=1.2)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow('w',out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# import sys
# sys.exit(0)

###########################################

## start training #######################

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
############################################



#### inference #######
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


# # ##### predictions on the test set in order to evaluate detector #######
dataset_dicts = get_ant_dicts(image_path, annotations_path, test_json)
# dictionary containing boxes and scores per image in order to calculate PR curves
preds_test = {}

for k,d in enumerate(dataset_dicts):
	im = cv2.imread(d["file_name"])
	outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
	# write this to a file to do PR curve
	preds = outputs['instances'].to('cpu')
	boxes = preds.pred_boxes
	scores = preds.scores
	num_boxes = np.array(scores.size())[0]
	list_of_boxes = []
	for i in range(0, num_boxes):
		coords = boxes[i].tensor.numpy()    	
		score = float(scores[i].numpy())
		list_of_boxes.append([int(coords[0][0]), int(coords[0][1]), int(coords[0][2]), int(coords[0][3]), score])

	preds_test[d["file_name"]] = list_of_boxes
	#v = Visualizer(im[:, :, ::-1],metadata=ant_metadata, scale=1.2)
	#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	#cv2.imwrite('/home/tarun/caltech-ee148/project/predictions/'+str(k)+'.png', out.get_image()[:, :, ::-1])

## write predictions to JSON file in order to plot PR curve #######
with open('/home/tarun/caltech-ee148/project/output_files/preds_test_manually_annotated.json','w') as f:
    json.dump(preds_test,f)


import sys
sys.exit(0)

### predictions on a given video(s) ####

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

list_of_videos = ['2021-05-13_18_01_00', '2021-05-13_06_01_00', '2021-05-13_07_01_00', '2021-05-13_08_00_59', '2021-05-13_09_00_59', '2021-05-13_10_00_59', '2021-05-13_11_01_00', '2021-05-13_12_01_00', '2021-05-13_13_01_00', '2021-05-13_14_01_00', '2021-05-13_15_00_59', '2021-05-13_19_00_59', '2021-05-13_19_31_00']
#list_of_videos = ['2021-05-13_09_00_59']

for vid_folder in list_of_videos:
	print (vid_folder)
	#vid_folder = '2021-05-13_09_00_59'
	parent_folder = '/home/tarun/Downloads/ant-data/beer-tree/'
	#parent_folder = '/home/tarun/Downloads/ant-data/courtyard/'
	#cap = cv2.VideoCapture(parent_folder+vid_folder+"/vid.h264")
	#total_frames = count_frames_manual(cap)
	#cap.release()
	cap = cv2.VideoCapture(parent_folder+vid_folder+"/vid.h264")

	print ('total frames : ' + str(total_frames))

	fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	vid_out = cv2.VideoWriter(parent_folder + 'predictions/' + vid_folder + '.mp4',fourcc, 30.0, (640,480))



	frame_number = 0
	predictions = {}

	while(1):

		if frame_number > 3400:
			break
		ret, frame = cap.read()
		#### preprocess
		#frame = frame[:, 50:-50, :]
		frame  = cv2.resize(frame, (640,480))

		boxes_color = []
		boxes_bg = []

		# histogram EQ - seems like this helps in dark cases but makes things slightly worse when there is lighting changes
		#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
		#frame2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		frame2 = frame

		outputs = predictor(frame2)
		preds = outputs['instances'].to('cpu')
		boxes = preds.pred_boxes
		scores = preds.scores
		num_boxes = np.array(scores.size())[0]
		list_of_boxes = []
		for i in range(0, num_boxes):
			coords = boxes[i].tensor.numpy()    	
			score = float(scores[i].numpy())
			list_of_boxes.append([int(coords[0][0]), int(coords[0][1]), int(coords[0][2]), int(coords[0][3]), score])
			frame2 = cv2.rectangle(frame2,(int(coords[0][0]),int(coords[0][1])),(int(coords[0][2]),int(coords[0][3])),(0,255,0),1)


		predictions[frame_number] = list_of_boxes
		v = Visualizer(frame2[:, :, ::-1],metadata=ant_metadata, scale=1.0)
		out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		
		#cv2.imshow('w', frame2)
		
		#cv2.imshow('w', out.get_image()[:, :, ::-1])
		#cv2.waitKey(30)
		vid_out.write(frame2)
		frame_number += 1


	with open(parent_folder + 'predictions/' + vid_folder + '.json','w') as f:
	    json.dump(predictions,f)

	cap.release()
	vid_out.release()
	cv2.destroyAllWindows()
