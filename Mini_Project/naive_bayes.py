import os 
from PIL import Image
import numpy as np
from array import *
import math
import sys

with open('sample_train.txt') as f:
    lines = f.read().splitlines()
# print(lines)
mlist = []

# print ( type(mlist))
for i in lines:
    words=list(i.split())  # Extract one line
    mlist.append(words)   # Now all files name( with name) is contained in the mlist
# print(mlist)
mlist.sort()

mat  = np.array(mlist) # Convert list to matrix
x 	 = mat[:,0]		# Slice the matrix (only 1st column)
x  	 = np.array(x).flatten() # Contain addresss
# print ( x)

mat  = np.array(mlist) 
y 	 = mat[:,1]
y  	 = np.array(y).flatten() # Contain sample names 'Alice,bob,abc'
# print ( type(mlist))
d = len(y)

dc = {}
j=1;
for i in y:  # dic[Alice] = 3 , dic[Bob]=6 , dic[Abc] = 9
	dc[i] = j
	j+=1;

#print(dc)

im_matrix = np.empty((d,1024))

j=0
for i in x:	
	img = np.array(Image.open(i).convert('L').resize((32, 32), Image.ANTIALIAS)) # Converting all images into grayscale & images to array
	# img_r = np.rimg,(64,64)) # Resize image
	pix = img.flatten()
	im_matrix[j] = pix  # Now creating a matrix to store all flatten images
	j+=1


im_mean = im_matrix.mean(0)  # 0 refers to column wise mean
im_matrix = im_matrix - im_mean    # Mean matrix
 
u,s,v = np.linalg.svd(im_matrix)

v = np.transpose(v); # Transpose
coeff = np.matmul(im_matrix,v)
coeff = coeff[:,:32] #300*32		 # Coefficient Matrix corresponding to sample image
# print(coeff)

##
with open('sample_test.txt') as f:
    lines = f.read().splitlines()

mlist_2 = []

for i in lines:
	words= list(i.split())  # Extract one line
	mlist_2.append(words)   # Now all files name( with name) is contained in the mlist

mat  = np.matrix(mlist_2)
a 	 = mat[:,0:1]
a  	 = np.array(a).flatten()

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
# print(coeff_2.shape)

mean = {}
var = {}

first = 0

# mean[Alice],mean[Bob],mean[abc] and similarly for var.
# Mistake  Var is going zero

# print(coeff)
for i in sorted(dc, key=dc.get): # Sort by value
	last = dc[i] # i : Alice & dc[i] = 3
	# print(first)
	mean[i] = np.mean(coeff[first:last, :], axis=0) # Axis=0 means column wise
	var[i] = np.var(coeff[first:last, :], axis=0)
	first = last

#print (mean)
# print (var)

#Now I have mean & var. for all classes

#x   from the Sample Image

e = 2.7183
pi = 3.142

maxi = -1000000
clas = []
pro  = []

for i in coeff_2:			# i is a row in coeff matrix of sample images, i.e i corresponds to a image
	maxi = -100
	for j in  dc:			# j represents class i.e alice,bob or abc
#		print(var[j])
		pro = ((i - mean[j])**2/(2*var[j]))
		pro = (e**((-1)*pro))
		pro = (pro/(2*pi*(var[j]))**(0.5))
		b = np.prod(pro)
		#print(b)
		
		if b > maxi:
			maxi = b
			z =j

	print (z) 
