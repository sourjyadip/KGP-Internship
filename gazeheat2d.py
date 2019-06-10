#Phani eyeglasses data 2d gaze point heatmap
import json
import matplotlib.pyplot as plt 
import numpy as np

data = [json.loads(line) for line in open('livedata.json', 'r')]
xlist=[]
ylist=[]

for i in range(10000):
	try:
		gaze=data[i]['gp']#'Gaze position'
		xlist.append(gaze[0])
		ylist.append(gaze[1])
	except KeyError:
		continue

x=np.array(xlist)
y=np.array(ylist)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=(64,64))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.title('2D Gaze point heatmap')
plt.ylabel('y')
plt.xlabel('x')
plt.xlim(0,1)
plt.ylim(0,1)
plt.imshow(heatmap, extent=[0,1,0,1])
plt.show()