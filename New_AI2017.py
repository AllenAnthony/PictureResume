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

# A:(408, 306)
# B:(372, 299, 3)
# C:(402, 266, 3)
# adjust based on 10
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal
# http://blog.csdn.net/pipisorry/article/details/49515215

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


samples = {'A':0.8, 'B':0.4, 'C':0.6}
for testName, noiseRatio in samples.iteritems():
    Img = im2double(imread(testName+'_ori.png'))
    Img[(Img==0)]=0.01 # distinguish the original black pixel and the corrupted pixel
    # imwrite(Img, testName+'_ori.png')
    corrImg = im2double(imread(testName+'.png'))
    ############### generate corrupted image ###############

    if len(Img.shape) == 2:
        Img = Img[:, :, np.newaxis]
        corrImg = corrImg[:, :, np.newaxis]
    rows, cols, channels = Img.shape

    ## generate noiseMask and corrImg
    '''
    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = round(noiseRatio * cols)
    for k in range(channels):n
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    corrImg = Img * noiseMask
    imwrite(corrImg, testName+'.png')
    '''

    noiseMask = np.array(corrImg!=0, dtype='double') # restore the moiseMask

    ## standardize the corrupted image
    minX = np.min(corrImg)
    maxX = np.max(corrImg)
    corrImg = (corrImg - minX)/(maxX-minX) # adjust the value of element to the ratio of the orignal ratio based on max ratio

    ## ======================learn the coefficents in regression function======================
    # In this section, we use gaussian kernels as the basis functions. And we
    # do regression analysis each row at a time.

    basisNum = 50 # number of basis functions
    sigma = 0.01 # standard deviation
    Phi_mu = np.linspace(1, cols, basisNum)/cols # mean value of each basis function, divide 1 into baisiNum parts, each ele is the upper-value
    Phi_sigma = sigma * np.ones((basisNum)) # set the standard deviation to the same value for brevity

    ## use pixel index as the independent variable in the regression function
    x = np.arange(cols)+1 #1-clos
    x = 1. * (x - np.min(x)) / (np.max(x)-np.min(x)) #divide 1 into clos-1 parts, and each element stand for the vlaue of segmentation

    resImg = np.copy(corrImg)

    for k in range(channels):
        for i in range(rows):
            ## select the missing pixels each row
            msk = noiseMask[i, :, k]
            misIdx = msk<1
            misNum = sum(misIdx)
            ddIdx = msk>=1
            ddNum = sum(ddIdx)

            ## compute the coefficients
            Phi = np.hstack((np.ones((ddNum, 1)), np.zeros((ddNum, basisNum-1)))) # the dimension is the normal pixel * number of function
            for j in range(1, basisNum): # 1-49
                Phi[:, j] = normpdf(x[(ddIdx)], Phi_mu[j-1], Phi_sigma[j-1]) * np.sqrt(2*np.pi) * Phi_sigma[j-1]
            w = dot(dot(pinv(dot(t(Phi), Phi)), t(Phi)), corrImg[(i,ddIdx,k)])

            ## restore the missing values
            Phi1 = np.hstack((np.ones((misNum, 1)), np.zeros((misNum, basisNum-1))))
            for j in range(1, basisNum):
                Phi1[:, j] = normpdf(x[(misIdx)], Phi_mu[j-1], Phi_sigma[j-1]) * np.sqrt(2*np.pi) * Phi_sigma[j-1]
            resImg[(i,misIdx,k)] = dot(Phi1, w)

    resImg = np.minimum(resImg, 1)
    resImg = np.maximum(resImg, 0)

    ## show the restored image
    if Img.shape[2] == 1: # grayscale image
        Img = Img.squeeze()
        corrImg = corrImg.squeeze()
        resImg = resImg.squeeze()
        plt.subplot(1, 3, 1)
        plt.imshow(Img, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(corrImg, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(resImg, cmap='gray')
        plt.show()
    else: # colorful image
        plt.subplot(1, 3, 1)
        plt.imshow(Img)
        plt.subplot(1, 3, 2)
        plt.imshow(corrImg)
        plt.subplot(1, 3, 3)
        plt.imshow(resImg)
        plt.show()

    prefix = 'result/%s_%.1f_%d'%(testName, noiseRatio, basisNum)
    fileName = prefix+'_result.h5'
    h5f = h5py.File(fileName,'w')
    h5f.create_dataset("basisNum", dtype='uint8', data=basisNum)
    h5f.create_dataset("sigma", dtype='double', data=sigma)
    h5f.create_dataset("Phi_mu", dtype='double', data=Phi_mu)
    h5f.create_dataset("Phi_sigma", dtype='double', data=Phi_sigma)
    h5f.create_dataset("resImg", dtype='double', data=resImg)
    h5f.create_dataset("noiseMask", dtype='double', data=noiseMask)
    h5f.close()
    #h5f = h5py.File(fileName,'r')
    #resImg = h5f.get('resImg').value
    #noiseMask = h5f.get('noiseMask').value
    #...
    #h5f.close()

    ## compute error
    im1 = Img.flatten()
    im2 = corrImg.flatten()
    im3 = resImg.flatten()
    print((
        '{}({}):\n'
        'Distance between original and corrupted: {}\n'
        'Distance between original and reconstructed (regression): {}'
    ).format(testName, noiseRatio, norm(im1-im2, 2), norm(im1-im3, 2)))

    ## store figure
    imwrite(resImg, prefix+'.png')