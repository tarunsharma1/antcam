#### helper script to read all.json and split it into train.json and val.json
import json
import random

with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_all.json') as f:
	data = json.load(f)

keys = list(data.keys())
random.shuffle(keys)

train_keys = keys[0:int(0.6*len(keys))]
val_keys = keys[int(0.6*len(keys)):int(0.8*len(keys))]
test_keys = keys[int(0.8*len(keys))::]


print (len(train_keys))
print (len(val_keys))
print (len(test_keys))

train_data = {}
val_data = {}
test_data = {}

for k in train_keys:
	train_data[k] = data[k]

for k in val_keys:
	val_data[k] = data[k]

for k in test_keys:
	test_data[k] = data[k]

# write the splits to json files
with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_train.json', 'w') as outfile:
	json.dump(train_data, outfile)

with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_val.json', 'w') as outfile:
	json.dump(val_data, outfile)

with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_test.json', 'w') as outfile:
	json.dump(test_data, outfile)

