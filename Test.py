# import pip
# for dist in pip.get_installed_distributions():
#     print(dist)

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

'''
A = imread('data\A.png')
print A.shape
info = np.iinfo(A.dtype)
print info.max
print "###################"
B = imread('data\B.png')
print B.shape
print B.dtype
info = np.iinfo(B.dtype)
print info.max
print "###################"
C = imread('data\C.png')
print C
print "###################"
print np.array(C!=0, dtype='double')
print C.shape
# list = [[1,2,3,4],
#         [2,3,4,5],
#         [3,4,5,6],
#         [4,5,6,7]]
# array=np.array(list)
# print array
# array[(array==3)]=0
# print array
#
# print array[:,:,np.newaxis]
# print array[:,np.newaxis]
# print array[np.newaxis]
#
# print np.max(array)
# print np.min(array)

# print np.ones((5))
# print np.linspace(1, 1000, 5)
# print "####################"
# print np.arange(100)
#
# print round(4.23764643,4)
# print np.random.permutation(9)
#
# print np.ones((5, 1))
# print np.zeros((5, 49))
# print "########################"
# print np.hstack((np.array([[1,2],[3,4]]), np.array([[9,8,7],[4,5,6]])))
# print range(1, 50)
# print "########################"
cols=10
a=np.array([1,0,1,0,1,0,1,0,1,0])
x = np.arange(cols) + 1  # 1-clos
x = 1. * (x - np.min(x)) / (np.max(x) - np.min(x))
print x
print x[5]
print (a==1)
print x[(a==1)]
# print "#####################"
# a=np.linspace(-1, 1, 21)
# print normpdf(a, 0, 1)
msk=np.array([1,0,1,0,1,0,1,0,1,0])
ddIdx = msk>=1
test=np.array([True,False,True,False,True,False,True,False,True,False])
print sum(ddIdx)
print x[(test)]
x[(test)][2]=0.989898988
print x

x[(test)]=0
print x

def fun(x,y):
    return x*y

print fun(np.array([[1,2,3],[1,1,1],[0,0,0]]),np.array([4,5,6]))

basisNum = 5 # number of basis functions
sigma = [[0.01, 0],
         [0, 0.01]]  # standard deviation
rows=3
cols=3
sigma = basisNum * [sigma]
Phi_sigma = np.array(sigma)
print sigma
print Phi_sigma
print Phi_sigma.shape
print Phi_sigma[0]
'''
# myrow=10
# mycol=10
#
# funRowNum=10
# funColNum=10
# basisNum = funRowNum*funColNum+1
# colarray=np.linspace(1, mycol, funColNum+1) / mycol
# rowarray=np.linspace(1, myrow, funRowNum+1) / myrow
# Phi_mu=np.zeros((funRowNum,funColNum,2))
#
# for i in xrange(0,funRowNum):
#     for j in xrange(0,funColNum):
#         Phi_mu[(i, j)]=[colarray[i],rowarray[j]]
# print Phi_mu
#
# print max(1,2,3,4,5)

a=range(0,30)
print a
a=np.array(a)
a=a.reshape((3,5,2))
print a
print "###############"
print a[1,1,1]
print a[0:2,0:3,1]
a=a.reshape((15,2))
print a
ddNum = sum(sum(a))
print ddNum
print "#################"
print "#################"
x=np.array([1,2,3])
y=x.copy()
print np.dot(x,y)
print np.ones((5))
# a=[0,1,2,0,3,0,6,0,5]
# a=np.array(a)
# a=a.reshape((3,3))
# ddMat = a > 0
# print ddMat
# print ddMat.flatten()

phi = np.hstack((np.ones((20, 1)), np.zeros((20, 10 - 1))))
print phi

shuju=np.array([1,2])
mu=np.array([1,2])
cov=np.array([[0.01,0],
              [0,0.01]])

print mulpdf.pdf(shuju,mu,cov)

stable=mulpdf(mu,cov)
print stable.pdf(shuju)

