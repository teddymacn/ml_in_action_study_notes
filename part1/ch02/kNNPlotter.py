"""
plot kNN figures

@author: Teddy.Ma
"""

from kNN import *

import matplotlib
import matplotlib.pyplot as plt

# 2.2.2 Figure 2.3 plot dating data simple
def plotDatingDataMat():
    datingDataMat,datingLabels = file2matrix2()
    fig = plt.figure("plotDatingDataMat")
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    plt.show()

# 2.2.2 Figure 2.4 plot dating data with different marker size and color
def plotDatingDataMat2():
    datingDataMat,datingLabels = file2matrix2()
    fig = plt.figure("plotDatingDataMat2")
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 
       15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

# 2.2.2 Figure 2.5 plot dating data with frequent flier miles vs % of time 
# spent playing video game
def plotDatingDataMat3():
    datingDataMat,datingLabels = file2matrix2()
    fig = plt.figure("plotDatingDataMat3")
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 
       15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

if __name__ == "__main__":
    plotDatingDataMat()
    plotDatingDataMat2()
    plotDatingDataMat3()
    