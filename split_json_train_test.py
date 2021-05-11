#### helper script to read all.json and split it into train.json and val.json
import json
import random

with open('/home/tarun/caltech-ee148/project/annotations/all.json') as f:
	data = json.load(f)

keys = list(data.keys())
random.shuffle(keys)

train_keys = keys[0:int(0.75*len(keys))]
val_keys = keys[int(0.75*len(keys))::]

train_data = {}
val_data = {}
for k in train_keys:
	train_data[k] = data[k]

for k in val_keys:
	val_data[k] = data[k]

# write the splits to json files
with open('/home/tarun/caltech-ee148/project/annotations/train.json', 'w') as outfile:
	json.dump(train_data, outfile)

with open('/home/tarun/caltech-ee148/project/annotations/val.json', 'w') as outfile:
	json.dump(val_data, outfile)

