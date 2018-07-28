# K-means Clustering

Chapter 10 talks about the first unsupervised learning algorithm in this book, the K-means Clustering algorithm.

K-means clustering is one of the most widely used unsupervised machine learning algorithms. It forms clusters of data based on the similarity between data instances. The only limitation of this algorithm is this algorithmn requires the the number of clusters, the K in K-means, to be defined beforehand.

This algorithm works like this: first, randomly pick k points from the dataset as centroids of each cluster, then iteratively performs three steps until none of the data points changes its cluster:

1. Find the Euclidean distance between each data point and centroids of all the clusters;
2. Assign the data points to the cluster of the centroid with nearest distance;
3. Calculate new centroids based on the mean values of the coordinates of all the data points from the corresponding cluster.

Given enough time, K-means will always converge, however similar to the problem of Gradien Descent in Polynomial Regression, it might converge to a local minimum rather than the global minimum. The result highly depends on the initial centroids it chose. Some enhancements to this issue include but not limit to:

- To run several times of K-means on the same dataset;
- To choose the initial centroids which are generally distant from each other;
- The idea of Bisecting k-means discussed in chapter 10.3, which starts out with one cluster and then splits the cluster in two, and so on;

## Demo Code

[kMeans.py](kMeans.py) - Revised version of the original kMeans demo