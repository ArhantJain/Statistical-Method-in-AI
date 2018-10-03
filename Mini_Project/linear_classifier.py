import os 
from PIL import Image
import numpy as np
from array import *
import math
import sys
from math import *


with open('./sample_train.txt') as f:
    lines = f.read().splitlines()
# print(lines)
mlist = []


for i in lines:
    words=list(i.split())  # Extract one line
    mlist.append(words)   # Now all files name( with name) is contained in the mlist
# print(mlist)
mlist.sort()
	
mat  = np.array(mlist) # Convert list to matrix
x 	 = mat[:,0]		# Slice the matrix (only 1st column)
x  	 = np.array(x).flatten() # Contain addresss


mat  = np.array(mlist) 
y 	 = mat[:,1]
y  	 = np.array(y).flatten() # Contain sample names 'Alice,bob,abc'

d = len(y) 	# no. of sample images i.e 9.

dc = {}
j=0;
for i in y:  # dic[Alice] = 2,(0,1,2) , dic[Bob]=5,(3,4,5) , dic[Abc] = 8,(6,7,8)
	dc[i] = j
	j+=1;

im_matrix = np.empty((d,1024)) # 9*1024

j=0
for i in x:	
	img = np.array(Image.open(i).convert('L').resize((32, 32), Image.ANTIALIAS)) # Converting all images into grayscale & images to array
	pix = img.flatten()
	im_matrix[j] = pix  # Now creating a matrix to store all flatten images
	j+=1


im_mean = im_matrix.mean(0)  # 0 refers to column wise mean
im_matrix = im_matrix - im_mean    # Mean matrix
 
u,s,v = np.linalg.svd(im_matrix)

v = np.transpose(v); # Transpose
coeff = np.matmul(im_matrix,v)
coeff = coeff[:,:32] 		 # Coefficient Matrix corresponding to sample image
n = coeff.shape[0]			 # No. of rows in coeff matrix

x1 = np.ones((n,1))
X = np.hstack((coeff,x1)) # Augmented Matrix 9*33
#print(X.shape)
# print(X)
d = len(dc) #  no. of classes
e = len(x) # e is the total no. of train images i.e 9

w = np.random.random((d,33)) # matrix of size =(no.of classes)*33
# print(w)


eta= 0.001

dc2 = {}
j=0;

for label in y:
	if label not in dc2:
		dc2[label]=j
		j+=1

for k in range(10):		# This loop will minimize the error in w
	for i in range(e): 		# e is no. of images
		const = -sys.maxsize -1
		a = 0
		p1 =0
		xt = np.transpose(X[i])
		# print(xt.shape)
		# print(y[i])
		power = np.matmul(w[dc2[y[i]]],xt)  # change
		for j in range(d):	# d is no. of classes
			p2 = np.matmul(w[j],xt)
			if p2>const:
				const = p2
		#print('b '+str(const))
		for j in range(d):	# d is no. of classes
			p2 = np.matmul(w[j],xt)
			a = a + exp(p2-const)  # Submission term in denominator

		q = exp(power-const)  # Ques. which w , ans: corresponding to the class of X[i]
		#print(a)
		p = q/a  				#Softmax classifier
		# print(p)
		w[dc2[y[i]]] = w[dc2[y[i]]] + eta*(1-p)*X[i]

# FINDING COEFFICIENT MATRIX FOR SAMPLE IMAGES

with open('sample_test.txt') as f:
    lines = f.read().splitlines()

mlist_2 = []

for i in lines:
	words= list(i.split())  # Extract one line
	mlist_2.append(words)   # Now all files name( with name) is contained in the mlist

mat  = np.matrix(mlist_2)
a 	 = mat[:,0:1]
a  	 = np.array(a).flatten() # a contain address of sample images

c = len(a)  # Length of list a
im_matrix_2 = np.empty((c,1024))

j=0
for i in a:	
	img = np.array(Image.open(i).convert('L').resize((32, 32), Image.ANTIALIAS)) # Converting all images into grayscale & images to array
	# img_r = np.resize(img,(64,64)) # Resize image
	pix = img.flatten()
	im_matrix_2[j] = pix  # Now creating a matrix to store all flatten images
	j+=1

im_matrix_2 = im_matrix_2 - im_mean    # im_mean is previously calculated
coeff_2 = np.matmul(im_matrix_2,v)     
coeff_2 = coeff_2[:,:32]				   # Coefficient matrix of test images
n = coeff_2.shape[0] # 5 Rows and 32 column
#print(coeff_2.shape) 

x1 = np.ones((n,1)) # 5*1
X2 = np.hstack((coeff_2,x1)) # Augmented matrix

for i in range(X2.shape[0]):
	s = np.transpose(X2[i])
	ans = np.matmul(w,s)
	ma  = ans.max()
	index = list(ans).index(ma)
	print(list(dc)[index])