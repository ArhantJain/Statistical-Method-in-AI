import matplotlib.image as mpimg
import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dirListing = os.listdir('./dataset') # Access the directory
editFiles = []
for item in dirListing:
   if ".jpg" in item:
       editFiles.append('./dataset' +'/'+item) # Saving path of images in list

#print (editFiles)

im_matrix = np.empty((520,1024))
j=0
for i in editFiles:	
	img = np.array(Image.open(i).convert('L')) # Converting all images into grayscale & images to array
	img_r = np.resize(img,(32,32)) # Resize image
	pix = img_r.flatten()
	im_matrix[j] = pix  # Now creating a matrix to store all flatten images
	j+=1



num_data,dim = im_matrix.shape # Gives Rows and columns

im_mean = im_matrix.mean(0)
im_matrix = im_matrix - im_mean

u,s,v = np.linalg.svd(im_matrix)

v = np.transpose(v); # Transpose
coeff = np.matmul(im_matrix,v) #Coefficient Matrix
a = coeff[:,0:1]
b = coeff[:,1:2]
c = coeff[:,2:3]
d = np.zeros(520)

x_axis = []
y_axis = []

'''for k in range(1,500,30):
	im_co   = np.matmul(coeff[:,:k],v[:,:k].T) # Image co-ordinate
	error= np.sum(np.multiply(im_co - im_matrix,im_co - im_matrix))
	x_axis.append(k)
	y_axis.append(error)'''

for i in range(100,600,10):
	# v = v[:,:i] # Cropped Eigen Vector
	i2 = np.matmul(coeff[:,:i],v[:,:i].T)
	diff = ( i2 - im_matrix)
	n1 = np.linalg.norm(diff, axis=1)
	n2 = np.linalg.norm(im_matrix, axis=1)
	ratio = n1/n2
	avg = np.mean(ratio)
	x_axis.append(i)
	y_axis.append(avg)


plt.scatter(a,d)
plt.xlabel('PCA-1')
plt.title('1-Dimensional Plot')

plt.figure()
plt.scatter(a,b)
plt.xlabel('PCA-2')
plt.ylabel('PCA-3')
plt.title('2-Dimensional Plot')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(a, b, c, c='r', marker='o')

ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.set_zlabel('PCA-3')
plt.title('3-Dimensional Plot')

plt.figure()
plt.plot(x_axis, y_axis)

plt.xlabel('No. of principal components')
plt.ylabel('Mean square error') 
plt.title('Mean square error vs No. of principal components')

# function to show the plot

plt.show()