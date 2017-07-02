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


samples = {'huidu':0.8,'I':0.4, 'J':0.6}
for testName, noiseRatio in samples.iteritems():

    corrImg = im2double(imread('data/'+testName+'.png'))
    print testName,":", noiseRatio
    if len(corrImg.shape) == 2:
        corrImg = corrImg[:, :, np.newaxis]
    rows, cols, channels = corrImg.shape

    noiseMask = np.array(corrImg!=0, dtype='double') # restore the moiseMask

    ## standardize the corrupted image
    minX = np.min(corrImg)
    maxX = np.max(corrImg)
    corrImg = (corrImg - minX)/(maxX-minX) # adjust the value of element to the ratio of the orignal ratio based on max ratio

    ## ======================learn the coefficents in regression function======================
    # In this section, we use gaussian kernels as the basis functions. And we
    # do regression analysis each row at a time.
    rowside=20
    colside=20
    myrow=rowside*2+1
    mycol=colside*2+1

    funRowNum = 10
    funColNum = 10
    basisNum = funRowNum*funColNum+1 # number of basis functions

    sigma = [[0.01, 0],
             [0, 0.01]]  # standard deviation
    sigma = (basisNum-1) * [sigma]
    Phi_sigma = np.array(sigma)# set the standard deviation to the same value for brevity basisNum*2*2

    rowarray = np.linspace(1, myrow, funRowNum + 1) / myrow
    colarray = np.linspace(1, mycol, funColNum + 1) / mycol
    Phi_mu = np.zeros((funRowNum, funColNum, 2))

    for i in xrange(0, funRowNum):
        for j in xrange(0, funColNum):
            Phi_mu[i,j] = [rowarray[i], colarray[j]] # mean value of each basis function, divide 1 into baisiNum parts, each ele is the upper-value

    Phi_mu = Phi_mu.reshape((funRowNum*funColNum,2))
    resImg = np.copy(corrImg)
    print "Phi_mu: ",Phi_mu.shape
    print "Phi_sigma: ",Phi_sigma.shape

    ## use pixel index as the independent variable in the regression function
    x=np.zeros((myrow,mycol,2))
    for i in xrange(0,myrow):
        for j in xrange(0,mycol):
            x[i,j]=[(1.0*i)/(myrow-1),(1.0*j)/(mycol-1)] # divide 1 into clos-1 parts, and each element stand for the vlaue of segmentation
    x = x.reshape((myrow*mycol,2))
    print "x: ",x.shape

    myposition=np.array([(1.0*rowside)/(myrow-1),(1.0*colside)/(mycol-1)])

    mypdf=[]
    mypdf.append(None)
    for couf in xrange(1,basisNum):
        mypdf.append(mulpdf(Phi_mu[couf - 1], Phi_sigma[couf - 1]))


    for i in xrange(0,rows):
        for j in xrange(0,cols):
            for k in xrange(0,channels):
                if corrImg[(i,j,k)]!=0:
                    continue
                else:
                    feild=np.zeros((myrow,mycol))
                    # rowbegin=max(i-rowside,0)
                    # rowend=min(i+rowside,rows-1)
                    # colbegin=max(j-colside,0)
                    # colend=min(j+colside,cols-1)
                    for cour in xrange(-rowside,rowside+1):
                        for couc in xrange(-colside,colside+1):
                            if( 0<=(i+cour)<rows and 0<=(j+couc)<cols):
                                feild[rowside+cour,colside+couc]=corrImg[i+cour,j+couc,k]

                    misMat = feild == 0
                    misNum = sum(sum(misMat))
                    ddMat = feild > 0
                    ddMat=ddMat.flatten()
                    ddNum = sum(ddMat)

                    ## compute the coefficients
                    phi = np.hstack((np.ones((ddNum, 1)), np.zeros((ddNum, basisNum - 1))))  # the dimension is the normal pixel * number of function
                    for couf in range(1, basisNum):  # 1-49
                        phi[:, couf] = mypdf[couf].pdf(x[(ddMat)]) * 2 * np.pi * np.sqrt(np.linalg.det(Phi_sigma[couf - 1]))

                    w = dot(dot(pinv(dot(t(phi), phi)), t(phi)), feild.flatten()[ddMat.flatten()])

                    ## restore the missing values
                    phi1=np.hstack((np.ones((1)) , np.zeros((basisNum-1))))
                    for couf in xrange(1,basisNum):
                        phi1[couf] = mypdf[couf].pdf(myposition) * 2 * np.pi * np.sqrt(np.linalg.det(Phi_sigma[couf - 1]))

                    resImg[i,j,k]=dot(phi1,w)

                    print "pixel: ",i*cols+j," k: ",k

    resImg = np.minimum(resImg, 1)
    resImg = np.maximum(resImg, 0)

    if corrImg.shape[2] == 1: # grayscale image
        corrImg = corrImg.squeeze()
        resImg = resImg.squeeze()

    # ## show the restored image
    # if corrImg.shape[2] == 1: # grayscale image
    #     corrImg = corrImg.squeeze()
    #     resImg = resImg.squeeze()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(corrImg, cmap='gray')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(resImg, cmap='gray')
    #     plt.show()
    # else: # colorful image
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(corrImg)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(resImg)
    #     plt.show()

    prefix = 'result20/%s_%.1f_%d'%(testName, noiseRatio, basisNum)

    ## store figure
    imwrite(resImg, 'data/'+prefix+'.png')