#### reads ant data after detection and after tracking and makes plots of ant count vs time, vs temparture, vs humidity
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json

mean_counts = {}
max_counts = {}

files = glob.glob('/home/tarun/Downloads/ant-data/beer-tree/tracking/*.txt')
for file in files: 
	f = open(file, 'r')
	data = f.readlines()
	# for every frame, i need the number of boxes
	counts_per_frame = {}
	for line in data:
		# remove the \n
		line = line[:-1]
		frame,ant_id,x1,y1,x2,y2,_ = [int(float(x)) for x in line.split(',')]
		
		# I had stopped the detections at frame 3400
		if frame >= 3400:
			continue

		if frame not in counts_per_frame.keys():
			counts_per_frame[frame] = 0

		# increment ant count for that frame
		counts_per_frame[frame] += 1

	
	# divide by number of frames - 3400 (i stopped detector at 3400)
	mean_counts[file] = np.sum(np.array(list(counts_per_frame.values())))/3400.0
	max_counts[file] =  np.max(np.array(list(counts_per_frame.values())))

# plot count vs time
files = list(mean_counts.keys())
files.sort()
x_labels = []
y_values_mean = []
y_values_max = []
temperature = []
humidity = []
lux = []

for file in files:
	x_labels.append(file.split('/')[-1][11:16])
	y_values_mean.append(mean_counts[file])
	y_values_max.append(max_counts[file])
	# open the corresponding data.txt file
	k = open('/home/tarun/Downloads/ant-data/beer-tree/' + file.split('/')[-1][:-4] + '/data.txt')
	data = json.load(k)

	temperature.append(data['temperature'])
	humidity.append(data['humidity'])
	lux.append(data['light']['lux'])


plt.plot(range(0,len(files)), y_values_mean, c='g', label='mean ant count')
plt.plot(range(0,len(files)), y_values_max, c='r', label='max ant count')

plt.plot(range(0,len(files)), temperature, c='b', alpha=0.2, label='temperature')
plt.plot(range(0,len(files)), humidity, c='m', alpha=0.2, label='humidity')
plt.plot(range(0,len(files)), lux, c='k', alpha=0.2, label='light level (lux)')


plt.title('ant count vs time of day')
plt.xticks(range(0,len(files)), x_labels, rotation=45)
plt.legend()
plt.show()


