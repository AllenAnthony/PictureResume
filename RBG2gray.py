# transform the RGB picture to gray picture

import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.stats as stats
import matplotlib.pyplot as plt

dot = np.dot

img = imread('data/gray_ori.png')

grayImg=np.zeros((img.shape[0],img.shape[1]),np.uint8)

for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        grayImg[row,col]=np.dot(img[row,col,0:3],[0.299,0.587,0.144])

## show the gray image
if len(grayImg.shape) == 2: # grayscale image
    plt.subplot(1, 1, 1)
    plt.imshow(grayImg, cmap='gray')
    plt.show()
else: # colorful image
    plt.subplot(1, 1, 1)
    plt.imshow(grayImg)
    plt.show()

print grayImg.shape
imsave('data/huidu_ori.png', grayImg)




