#### helper script to read all.json and split it into train.json and val.json
import json
import random
import numpy as np

with open('/home/tarun/caltech-ee148/project/annotations/manually_annotated_all.json') as f:
	data = json.load(f)

keys = list(data.keys())


train_data = {}
val_data = {}
test_data = {}

## we want to make sure all the "hard" annotations (2 frames each from 2 morning videos) are in the training set
special_idxs = np.where(np.array([ '2021-05-13_10' in keys[i].split('/')[-1] for i in range(0,len(keys))])==True)[0]
for i in special_idxs:
	print (keys[i])
	train_data[keys[i]] = data[keys[i]]

special_idxs = np.where(np.array([ '2021-05-13_07' in keys[i].split('/')[-1] for i in range(0,len(keys))])==True)[0]
for i in special_idxs:
	print (keys[i])
	train_data[keys[i]] = data[keys[i]]

# exclude the above keys before the split
keys = [j for j in keys if (j not in list(train_data.keys()))]

random.shuffle(keys)

train_keys = keys[0:int(0.6*len(keys))]
val_keys = keys[int(0.6*len(keys)):int(0.8*len(keys))]
test_keys = keys[int(0.8*len(keys))::]


print (len(train_keys))
print (len(val_keys))
print (len(test_keys))


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

