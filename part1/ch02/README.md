# k-Nearest Neighbors

kNN is the first machine learning algorithm dicussed in book "Machine Learning in Action", which is a very good choice as a starting point, coz it is simple enough for a person even without or forgot advanced Maths knowledge to understand it.

This chapter demonstrates only the kNN algorithm for classification, but I'd like to extend it a little bit.

k-Nearest Neighbors is one of the neighbors based algorithms. It is most commonly used in supervised learning. When working with discrete data, it is called "k-Nearest Neighbors Classification", when working with continuous data, it is called "k-Nearest Neighbors Regression".

The basic idea of k-Nearest Neighbors is: it does not attempt to construct a general internal model, but simply stores instances of the training data. A simplest implementation considers the majority vote of each of the k nearest neighbors to be with a uniform weight, which is what we do in the demo code of this chapter. Another option, which works better in some circumstances, is to weight the majority votes of nearer neighbors more than those of the far away ones.

## Variants

The variants of neighbors based algorithms are mainly different in how to choose points and how many points to choose to compare the distance with.

In the demo code of this chapter, when searching points to choose the k nearest neighbors, we are using a simple brute force algorithm, meaning to compare distance of each point pair in the training data. It's cost is O(DN), where D means number of dimentionality and N means number of points. 

One common approach to enhance performance is to reduce the point search area so that to reduce the number of point distance calculation, which is to use a fixed radius R to only search points inside the this area. This algorithm is called "Fixed-radius Nearest Neighbors". But for high-dimensional parameter spaces, this method becomes less effective due to the so-called "[curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)".

There are other common point search algorithms: Ball Tree and k-d Tree. These two point search algorithms try to reduce search cost. Ball Tree's cost is around O[Dlog(N)], while k-d Tree's is between O(DN) and O[Dlog(N)], when D is small (D<20), it's close to O[Dlog(N)], and when D is large, it's close to O(DN).

Although Ball Tree and k-d Tree might have smaller search cost, it doesn't mean they are always better in general. Firstly, Ball Tree and k-d Tree's query time will become slower as k increases. Secondly, both Ball Tree and k-d Tree require a construction phase. If N is relatively small, the cost of the construction phase might be larger than the total search cost they can save.

*Additional [reference](http://scikit-learn.org/stable/modules/neighbors.html#choice-of-nearest-neighbors-algorithm) for more trade-offs about how to choose an optimal point search algorithm.*

## Demo Code

[kNN.py](kNN.py) - Revised version of the original kNN demo

[kNNSklearn.py](kNNSklearn.py) - A scikit-learn version of kNN implementation

[kNNPlotter.py](kNNPlotter.py) - The scripts for plotting figures in this chapter