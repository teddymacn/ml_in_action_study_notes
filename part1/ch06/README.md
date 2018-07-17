# Support Vector Machine

Chapter 6 will be the most tough algorithm so far in this book.

Support Vector Machine is one of the most powerful and widely used algorithms in machine learning. It is as important as neural networks, and in many scenarios, it is even better. The idea of Support Vector Machine is not hard to understand, it's optimization objective is to maximize the margin between the separator and the closest points to the separator. The closest points are called "support vectors", that's where the name "Support Vector Machine" comes from.

Why we say it is the most tough algorithm so far is coz the algorithms to implement SVM requires quite advanced Maths knowledge in linear algebra. In the demo code, you will see quite heavy linear algebra computations. So to understand the code, you might have to review your linear algebra knowledge and to get some practice in python numpy matrix computations first.

The main benefits of SVM are:
- It is effective in high dimensional spaces
- It is still effective in cases where number of dimensions is greater than the number of samples
- It is memory efficient
- It is versatile by specifying different kernel functions

## Demo Code

[svm.py](svm.py) - Revised version of the original svm demo

[svmSklearn.py](svmSklearn.py) - A scikit-learn version of svm implementation

[svmPlotter.py](svmPlotter.py) - The scripts for plotting figures in this chapter
