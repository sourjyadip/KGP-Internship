import json
from collections import defaultdict
import pandas as pd

dlist=[]
data = [json.loads(line) for line in open('livedata.json', 'r')]
for i in range(10000):
	
	if(data[i]['s']==0):
		d={}
		
		ts=data[i]['ts']
		d['time']=ts
		try:
			d['pupildiam']=data[i]['pd']#'Pupil Diameter'
		except KeyError:
			#pass
			d['pupildiam']=' ' 
		try:
			d['gazepos']=data[i]['gp']#'Gaze position'
		except KeyError:
			d['gazepos']=' '
		try:
			d['gaze3d']=data[i]['gp3']#'Gaze position 3d'
		except KeyError:
			d['gaze3d']=' '
		try:
			d['gazedir']=data[i]['gd']#'Gaze direction'
		except KeyError:
			d['gazedir']=' '
		try:
			d['pcentre']=data[i]['pc']#Pupil centre
		except KeyError:
			d['pcentre']=' '
		dlist.append(d)

for i in range(100):
	print(dlist[i])