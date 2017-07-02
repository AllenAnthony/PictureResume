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



samples = {'huidu': 0.8, 'I': 0.4, 'J': 0.6}
for testName, noiseRatio in samples.iteritems():
    for side in [3, 5, 7, 10, 13, 16, 20]:
        Img = im2double(imread('data/' + testName + '_ori.png'))
        Img[(Img == 0)] = 0.01  # distinguish the original black pixel and the corrupted pixel
        corrImg = im2double(imread('data/' + testName + '.png'))
        resfilename = 'data/result%d/%s_%0.1f_101.png' % (side, testName, noiseRatio)
        resImg = im2double(imread(resfilename))

        ############### generate corrupted image ###############

        if len(Img.shape) == 2:
            Img = Img[:, :, np.newaxis]
        if len(corrImg.shape) == 2:
            corrImg = corrImg[:, :, np.newaxis]
        if len(resImg.shape) == 2:
            resImg = resImg[:, :, np.newaxis]

        ## compute error
        Img = Img.flatten()
        corrImg = corrImg.flatten()
        resImg = resImg.flatten()
        print((
                  '{}:\n'
                  '{}({}):\n'
                  'Distance between original and corrupted: {}\n'
                  'Distance between original and reconstructed (regression): {}\n'
              ).format(side, testName, noiseRatio, norm(Img - corrImg, 2), norm(Img - resImg, 2)))



