# Support Vector Machine

Chapter 6 will be the most tough algorithm in this book.

Support Vector Machine is one of the most powerful and widely used algorithms in machine learning. It is as important as neural networks, and in many scenarios, it is even better. The idea of Support Vector Machine is not hard to understand, it's optimization objective is to maximize the margin between the separator and the closest points to the separator. The closest points are called "support vectors", that's where the name "Support Vector Machine" comes from.

Why we say it is the most tough algorithm in this book is coz algorithms to implement SVM require quite a lot advanced Maths knowledge, especially the optimization theory. In the demo code, you will see quite a lot equation transformations, derivatives and linear algebra computations. So to understand the code, you might have to review related Maths knowledge first.

Two materials are recommended to read first which step-by-step explains mothe Math details behind SVMs: [svm-tutorial](https://www.svm-tutorial.com/svm-tutorial/) by 
Alexandre KOWALCZYK and [CS229 Lecture notes](http://cs229.stanford.edu/notes/cs229-notes3.pdf) by Andrew Ng.

When doing classification on simple linearly separable datasets, the only strength of SVM is its memory efficiency.

What really makes SVM be so important and attractive against other classification algorithms is how it works with non-linearly separable datasets. Different from those regression based tools such as [polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression), which tries to fit a higher order equation and requires exponentially growing heavey computation, the idea in SVM, called the kernel trick, is, as long as you can find a suitable kernel function, you can map your non-linear dataset from a lower dimension to a higher dimension, so that you can do cheap linear classification in a higher dimension to solve the original non-linear classification problem in the lower dimension.

In the demo code:
- The smoSimple algorithm works for linear classification on small dataset;
- The smoP algorithm works together with radial bias function, which can map 2D data to 3D to do higher dimensional linear classification, works for a non-linearly separable dataset;

## Demo Code

[svm.py](svm.py) - Revised version of the original svm demo

[svmSklearn.py](svmSklearn.py) - A scikit-learn version of svm implementation

[svmPlotter.py](svmPlotter.py) - The scripts for plotting figures in this chapter

[svm_rewritten.py](svm_rewritten.py) - Runnable all-in-one rewritten version
