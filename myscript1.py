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

for d in ["train", "val"]:
    DatasetCatalog.register("ant_" + d, lambda d=d: get_ant_dicts('/home/tarun/caltech-ee148/project/images', '/home/tarun/caltech-ee148/project/annotations', d))
    MetadataCatalog.get("ant_" + d).set(thing_classes=["ant"])

ant_metadata = MetadataCatalog.get("ant_train")



cfg = get_cfg()
cfg.merge_from_file("/home/tarun/caltech-ee148/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
#cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
cfg.DATASETS.TRAIN = ("ant_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = "/home/tarun/Downloads/model_final_721ade.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITERS = 20000    # 10 epochs gave me 18000 iterations...that is 1800 iterations per epoch..i think 2 images per batch and 2 workers? 
cfg.SOLVER.STEPS = [10000, 15000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
# trainer.train()


#### inference on val set
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset_dicts = get_ant_dicts('/home/tarun/caltech-ee148/project/images', '/home/tarun/caltech-ee148/project/annotations', 'val')
for k,d in enumerate(dataset_dicts):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=ant_metadata, 
                   scale=1.2,    # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('/home/tarun/caltech-ee148/project/predictions/'+str(k)+'.png', out.get_image()[:, :, ::-1])