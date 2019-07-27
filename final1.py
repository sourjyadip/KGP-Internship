'''
Stuff to change every image:
Size of excel dataset n1
Image name
Slide name
Dimensions for freuency tuned saliency detection
'''
import numpy as np 
import math
from scipy import stats
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.util import img_as_float
from scipy.optimize import minimize
import cv2
import pandas as pd 
import json

def quadct(x,y,xx,yy,n):
	na=0
	nb=0
	nc=0
	nd=0
	for k in range(0,n-1):
		if yy[k]>y:
			if xx[k]>x:
				na=na+1
			else:
				nb=nb+1
		else:
			if xx[k]>x:
				nd=nd+1
			else:
				nc=nc+1
	ff=1.0/n
	fa=ff*na
	fb=ff*nb
	fc=ff*nc
	fd=ff*nd
	return fa,fb,fc,fd

def probks(alam):
	fac = 2.0
	sum = 0.0
	termbf = 0.0

	a2 = -2.0*alam*alam
	for j in range(1,101):#change if it doesn't converge
		term = fac*math.exp(a2*j*j)
		sum += term
		if math.fabs(term) <= 0.001*termbf or math.fabs(term) <= 1.0e-8*sum:
			return sum
		fac = -fac
		termbf = math.fabs(term)

	return 1.0 #failed to converge



def ks2d(x1,y1,n1,x2,y2,n2):
	d1=0.0
	for j in range(0,n1-1):
		fa,fb,fc,fd=quadct(x1[j],y1[j],x1,y1,n1)
		ga,gb,gc,gd=quadct(x1[j],y1[j],x2,y2,n2)
		d1=max(abs(fa-ga),abs(fb-gb),abs(fc-gc),abs(fd-gd))

	d2=0.0
	for j in range(0,n1-1):
		fa,fb,fc,fd=quadct(x2[j],y2[j],x1,y1,n1)
		ga,gb,gc,gd=quadct(x2[j],y2[j],x2,y2,n2)
		d1=max(abs(fa-ga),abs(fb-gb),abs(fc-gc),abs(fd-gd))

	d=0.5*(d1+d2)
	squen=math.sqrt(float(n1*n2)/float(n1+n2))

	r1,p1=stats.pearsonr(x1,y1)
	r2,p2=stats.pearsonr(x2,y2)
	rr=math.sqrt(1.0-0.5*(r1**2+r2**2))
	prob=probks(d*squen/(1.0+rr*(0.25-0.75/squen)))
	return prob 

def threshval(img): #finds treshold for binarization
	width = np.size(img,1)
	height = np.size(img,0)
	c=0
	val=0
	for i in range(0,height):
		for j in range(0,width):
			c=c+img[i,j]
	val=int(c/(width*height))
	return val

def totalpoints(img):#Finds total number of salient points or gaze point ground truth data
	width = np.size(img,1)
	height = np.size(img,0)
	c=0
	for i in range(0,height):
		for j in range(0,width):
			if img[i,j]==255: #highest value is 250
				c=c+1
	return c

def getpoints(img):#gets points from 1440x900 saliency map
	width = np.size(img,1)
	height = np.size(img,0)
	xlist=[]
	ylist=[]
	for i in range(0,height):
		for j in range(0,width):
			if img[i,j]==255:
				xlist.append(j)
				ylist.append(i)
				#print j,i
	return xlist,ylist

def fretune(img):#Builds the saliency map

	img_rgb = img_as_float(img)

	img_lab = skimage.color.rgb2lab(img_rgb) 

	mean_val = np.mean(img_rgb,axis=(0,1))

	kernel_h = (1.0/16.0) * np.array([[1,4,6,4,1]])
	kernel_w = kernel_h.transpose()

	blurred_l = scipy.signal.convolve2d(img_lab[:,:,0],kernel_h,mode='same')
	blurred_a = scipy.signal.convolve2d(img_lab[:,:,1],kernel_h,mode='same')
	blurred_b = scipy.signal.convolve2d(img_lab[:,:,2],kernel_h,mode='same')

	blurred_l = scipy.signal.convolve2d(blurred_l,kernel_w,mode='same')
	blurred_a = scipy.signal.convolve2d(blurred_a,kernel_w,mode='same')
	blurred_b = scipy.signal.convolve2d(blurred_b,kernel_w,mode='same')

	im_blurred = np.dstack([blurred_l,blurred_a,blurred_b])

	sal = np.linalg.norm(mean_val - im_blurred,axis = 2)
	sal_max = np.max(sal)
	sal_min = np.min(sal)
	sal = (255 * ((sal - sal_min) / (sal_max - sal_min))).astype("uint8")

	return sal

def getexdata():
	xls = pd.ExcelFile('Phani_Trial.xlsx')
	df1 = pd.read_excel(xls, 'Sheet6') #change
		
	xlist=[]
	ylist=[]
	dx1 = pd.DataFrame(df1, columns= ['GazePosX'])
	dy1 = pd.DataFrame(df1, columns= ['GazePosY'])

	for i in range(0,350):#change
		if math.isnan(dx1.iloc[i]['GazePosX'])==False:
			xlist.append(int(dx1.iloc[i]['GazePosX']))
			ylist.append(int(dy1.iloc[i]['GazePosY']))

	return xlist,ylist,len(xlist) #change


def getsaldata():#changed
	img = skimage.io.imread('tulips.png')#change
	img = cv2.resize(img,(1440,900), interpolation = cv2.INTER_CUBIC) #change dimensions
	#img = cv2.GaussianBlur(img,(5,5),0) #experimentally done
	sal=fretune(img)
	#sal = cv2.GaussianBlur(sal,(5,5),0) #experimentally done
	k=threshval(sal)
	ret,sal = cv2.threshold(sal,k+20,255,cv2.THRESH_BINARY)#threshold value increased by 20 gives good resuts experimentally
	x,y=getpoints(sal)#changed
	#x2,y2,n2=average(x,y,n1)
	n=totalpoints(sal)
	return x,y,n

def main():
	x1,y1,n1=getexdata()
	x2,y2,n2=getsaldata()
	p=ks2d(x1,y1,n1,x2,y2,n2)
	print p

if __name__ == '__main__':
	main()