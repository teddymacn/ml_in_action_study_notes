# Machine Learning in Action Study Notes

The purpose of this repo is to keep my notes when studying machine learning algorithms.

Chapters in this repo cover two books and are divided in two parts.

Part I - Study notes of the published book "[Machine Learning in Action](https://www.manning.com/books/machine-learning-in-action)".

This is a great book, introducing the basis of many common machine learning algorithms with demo code in python 2.X. To be fair, the code of this book is good enough as demos of each chapter. But as an audience who is an experienced developer like me, who want to apply these machine learning algorithms in potential real projects, I always want to revise the implementation of each algorithm in my own style, for example, be python 3.X compatible, with more practical libraries, with more managable code styles, and with some extension of my understanding, etc. In short, I want my revised code implmentation and my extension make these algorithms be more friendly for a developer to reference & apply.

Furthermore, for most of the algorithms mentioned in this book, there are many common general purpose implementations in many high level machine learning frameworks. So beside a revised version of each algorithm, I also try to provide an additional implementation for the same solution but with a high level machine learning framework such as [scikit-learn](http://scikit-learn.org), which might be a more practical way if we want to try on or apply these algorithms efficiently in real life.

Part II - Study notes of the free online book "[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)".

This online book covers the basis of neural networks and deep learning, which are very good supplements to Part I. The best part of this book is it does not focus on applications of neural networks algorithms, but it step-by-step introduces the evolution of neural networks towards deep learning, showing you the problems and common solutions during the evolution. So when finishing the study of this book, you should be confident enough to take look at other neural network applications, demos and frameworks, and be able to understand how to adjust neural network parameters and what the potential issues are.

And similar to the book of part 1, it not only introduces the neural networks algorithms step by step, but also provides the python implementations from scratch and also with detailed comments in the source code.

All the demos in this book are doing classification with the mnist handwriting dataset. The mnist dataset is so popular that you can easily find different implementations with different neural network frameworks such as tensorflow, theano, keras and pytorch. So I don't provide additional demos in the notes of this book like in part 1, you can just google by yourself if you have interests.

## Study Environment
- Spyder IDE v3.X with Python 3.X
- Libraries:
    - numpy - for common science computation
    - matplotlib - for simply scatter and line plotting
    - pandas - to simplify test data loading
    - pydotplus - for graphviz based plotting, e.g. tree plotting
    - sklearn - the high level machine learning framework to demo with
    - theano - for running neural network demos in GPU for part 2

## Chapters

Part I - Machine Learning in Action

- Supervised Learning
    - ch02 [k-Nearest Neighbors](./part1/ch02/README.md)
    - ch03 [Decision Trees](./part1/ch03/README.md)
    - ch04 [Naive Bayes](./part1/ch04/README.md)
    - ch05 [Logistic Regression](./part1/ch05/README.md)
    - ch06 [Support Vector Machine](./part1/ch06/README.md)
    - ch07 [AdaBoost](./part1/ch07/README.md)
    - ch08 [Linear Regression](./part1/ch08/README.md)
    - ch09 [Tree-based Regression](./part1/ch09/README.md)
- Unsupervised Learning
    - ch10 [K-means Clustering](./part1/ch10/README.md)
    - ch11 [Apriori](./part1/ch11/README.md)
    - ch12 [FP-growth](./part1/ch12/README.md)
- Dimensionality Reduction
    - ch13 [Principal Component Analysis (PCA)](./part1/ch13/README.md)
    - ch14 [Singular
Value Decomposition (SVD)](./part1/ch14/README.md)

Part II - Neural Networks and Deep Learning

- ch01 [Using neural nets to recognize handwritten digits](./part2/ch01.md)
- ch02 [How the backpropagation algorithm works](./part2/ch02.md)
- ch03 [Improving the way neural networks learn](./part2/ch03.md)
- ch05 [Why are deep neural networks hard to train?](./part2/ch05.md)
- ch06 [Deep learning](./part2/ch06.md)
