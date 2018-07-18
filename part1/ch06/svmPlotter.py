'''
Plotting svm Source Code for Machine Learning in Action Ch. 6

@author: Teddy Ma
'''

from svm import *

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numpy import *
from sklearn.svm import SVC

# get support vectors from smo output
def getSVs(dataArr, labelArr, alphas):
    datMat = mat(dataArr)
    svInd = nonzero(alphas.A>0)[0]
    sVs = datMat[svInd] #get matrix of only support vectors
    labelSV = array(labelArr)[svInd]
    return sVs, labelSV

# 6.3.2 Figure 6.4 plot SmoSimple on sample dataset
def plotSmoSimpleLinear():
    dataArr,labelArr = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataArr, labelArr, 200, 0.0001, 100) #C=200 important
    svs,labels = getSVs(dataArr, labelArr, alphas)
    w = calcWs(alphas, dataArr, labelArr)
    x = arange(-2.0, 12.0, 0.1)
    y = array((-w[0]*x - b)/w[1])[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    ax.axis([-2,12,-8,6])
    
    # plot label 1 points
    i0 = nonzero(mat(labelArr).A > 0)[1] # get index of label 1 points
    xy0 = array(dataArr)[i0,:]
    ax.scatter(xy0[:,0],xy0[:,1], marker='o', s=30, c='red')
    
    # plot label -1 points
    i1 = nonzero(mat(labelArr).A < 0)[1] # get index of label -1 points
    xy1 = array(dataArr)[i1,:]
    ax.scatter(xy1[:,0],xy1[:,1], marker='s', s=30)
    
    # plot separator line
    ax.plot(x,y)
    
    # plot support vectors
    for pt in array(svs):
        circle = Circle((pt[0], pt[1]), 0.4, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    plt.show()

# 6.3.2 Figure 6.4 plot Sklearn based linear svm on sample dataset
def plotSklearnLinear():
    dataArr,labelArr = loadDataSet('testSet.txt')
    clf = SVC(kernel='linear')
    clf.fit(array(dataArr), array(labelArr))
    b = array([clf.intercept_])
    svs,labels = array(mat(dataArr)[list(clf.support_),:]),array(mat(labelArr)[:,list(clf.support_)])
    w = clf.coef_[0]
    x = arange(-2.0, 12.0, 0.1)
    y = array((-w[0]*x - b)/w[1])[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    ax.axis([-2,12,-8,6])
    
    # plot label 1 points
    i0 = nonzero(mat(labelArr).A == 1)[1] # get index of label 1 points
    xy0 = array(dataArr)[i0,:]
    ax.scatter(xy0[:,0],xy0[:,1], marker='o', s=30, c='red')
    
    # plot label -1 points
    i1 = nonzero(mat(labelArr).A == -1)[1] # get index of label -1 points
    xy1 = array(dataArr)[i1,:]
    ax.scatter(xy1[:,0],xy1[:,1], marker='s', s=30)
    
    # plot separator line
    ax.plot(x,y)
    
    # plot support vectors
    for pt in array(svs):
        circle = Circle((pt[0], pt[1]), 0.4, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    plt.show()

# 6.5.3 Figure 6.8 plot SmoP with Rbf on rbf dataset
def plotSmoPRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    svs,labels = getSVs(dataArr, labelArr, alphas)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    ax.axis([-2,2,-1,1.2])
    
    # plot label 1 points
    i0 = nonzero(mat(labelArr).A == 1)[1] # get index of label 1 points
    xy0 = array(dataArr)[i0,:]
    ax.scatter(xy0[:,0],xy0[:,1], marker='o', s=30, c='red')
    
    # plot label -1 points
    i1 = nonzero(mat(labelArr).A == -1)[1] # get index of label -1 points
    xy1 = array(dataArr)[i1,:]
    ax.scatter(xy1[:,0],xy1[:,1], marker='s', s=30)
    
    # plot support vectors
    for pt in array(svs):
        circle = Circle((pt[0]+0.005, pt[1]-0.01), 0.05, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    plt.show()

# 6.5.3 Figure 6.8 plot Sklearn based Rbf on rbf dataset
def plotSklearnRbf():
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    clf = SVC()
    clf.fit(array(dataArr), array(labelArr))
    svs,labels = array(mat(dataArr)[list(clf.support_),:]),array(mat(labelArr)[:,list(clf.support_)])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    ax.axis([-2,2,-1,1.2])
    
    # plot label 1 points
    i0 = nonzero(mat(labelArr).A == 1)[1] # get index of label 1 points
    xy0 = array(dataArr)[i0,:]
    ax.scatter(xy0[:,0],xy0[:,1], marker='o', s=30, c='red')
    
    # plot label -1 points
    i1 = nonzero(mat(labelArr).A == -1)[1] # get index of label -1 points
    xy1 = array(dataArr)[i1,:]
    ax.scatter(xy1[:,0],xy1[:,1], marker='s', s=30)
    
    # plot support vectors
    for pt in array(svs):
        circle = Circle((pt[0]+0.005, pt[1]-0.01), 0.05, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    plt.show()
