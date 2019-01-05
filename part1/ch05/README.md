# Logistic Regression

In chapter 5, we encountered the first optimization algorithm, specifically the first regression algorithm: Logistic Regression.

Comparing to the previous chapters, this chapter discusses limited Maths behind regression, so if you have forgotten the Maths knowledge about linear algebra and derivative，it might be a little big hard to understand the gradient and the convergence procedure of regression.

Logistic Regression can be considered as a varient of Linear Regression for a logistic result. So a more natural learning path of regression might be to learn [Linear Regression](../ch08/README.md) first. The main difference of Logistic Regression from Linear Regression is, the word "Logistic" indicates that, for each feature, the prediction cares about only a probability value in range [0, 1], while Linear Regression's prediction result can be an arbitrary number value. Logistic Regression uses a logistic sigmoid function to convert an arbitrary value to a probablity value in range [0, 1] which can then be mapped to two or more discrete classes.

When training a logistic regression model, we usually use the gradient ascent algorithm or its variations. But this book is lack of explanation for why we should use gradient ascent rather than gradient descent. So it really took me some time to understand this not until I read [this paper](http://cs.wellesley.edu/~sravana/ml/logisticregression.pdf). Simply speaking, when using gradient ascent, we try to maximize something. And in logistic regression, what we try to maximize is the likelihood or total probability for the model to predict the correct class label of each example in the training dataset.

Andrew Ng has a very popular [machine learning course on coursera](https://www.coursera.org/learn/machine-learning), the first several chapters of which step-by-step introduce from Linear Regression to Logistic Regression with all the Maths knowledge behind them clearly, which is highly recommended if you meet any troubles in understanding this chapter.

Logistic Regression is one of the most important machine learning algorithms. It is the also the core component of neural networks and deep learning coz each node of a hidden or output layer in neural networks does kind of a Logistic Regression. So it is worth paying more attention on it and to fully understand it.

## Demo Code

[logRegres.py](logRegres.py) - Revised version of the original logRegres demo

[logRegresSklearn.py](logRegresSklearn.py) - A scikit-learn version of logRegres implementation

[logRegresPlotter.py](logRegresPlotter.py) - The scripts for plotting figures in this chapter

[logRegres_rewritten.py](logRegres_rewritten.py) - Runnable all-in-one rewritten version