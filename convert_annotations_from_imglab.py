#### annotations were done using browser based tool - imglab.in : https://github.com/NaturalIntelligence/imglab
#### annotations were downloaded from the tool in the COCO Json format
#### this scripts converts annotations from that format into the desired format for training

import json
import numpy as np
import os
from os import path

f= open('/home/tarun/Downloads/2_coco_imglab.json')
data = json.load(f)

images = data['images']

annotations = {}
for image in images:
	image_id = image['id']
	annotations['/home/tarun/caltech-ee148/project/images_beertree_vids/' + image['file_name']] = []
	idx = 0
	# go through annotations and pick out the ones with the same image_id as image
	for k in data['annotations']:
		if k['image_id'] == image_id:
			# I think this is top left X,Y,H,W
			b = k['bbox']
			# convert it to X,Y,X,Y
			annotations['/home/tarun/caltech-ee148/project/images_beertree_vids/' + image['file_name']].append({'key':idx,'bbox':[int(b[0]),int(b[1]),int(b[0]) + int(b[2]),int(b[1]) + int(b[3])]})
			idx += 1




### write all annotations (from multiple iterations of annotations) to json
if path.exists('/home/tarun/caltech-ee148/project/annotations/manually_annotated_all.json'):
	with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_all.json', 'r+') as outfile:
		data = json.load(outfile)
		data.update(annotations)
		outfile.seek(0)
		json.dump(data, outfile)
else:
	with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_all.json', 'w') as outfile:
		json.dump(annotations, outfile)

