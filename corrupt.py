import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.stats as stats
import matplotlib.pyplot as plt
import h5py

normpdf = stats.norm.pdf
norm = np.linalg.norm
pinv = np.linalg.pinv
dot = np.dot
t = np.transpose
mulpdf = stats.multivariate_normal


def im2double(im): #transform the value of RBG to ratio(float)
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max

def imwrite(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype==np.double:
        #img = np.array(img*255, dtype=np.uint8)
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imsave(filename, img)


samples = {'huidu':0.8}
for testName, noiseRatio in samples.iteritems():
    Img = im2double(imread('data/'+testName+'_ori.png'))
    Img[(Img==0)]=0.01 # distinguish the original black pixel and the corrupted pixel
    # imwrite(Img, testName+'_ori.png')
    ############### generate corrupted image ###############

    print Img.shape
    if len(Img.shape) == 2:
        Img = Img[:, :, np.newaxis]
    rows, cols, channels = Img.shape

    ## generate noiseMask and corrImg

    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = int(round(noiseRatio * cols))
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    corrImg = Img * noiseMask
    imwrite(corrImg, 'data/'+testName+'.png')