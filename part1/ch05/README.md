# Logistic Regression

In chapter 5, we encountered the first optimization algorithm, specifically the first regression algorithm: Logistic Regression.

Frankly speaking, comparing to the previous chapters, this chapter discusses too less about the related Maths behind regression, so if you have limited Maths knowledge about linear algebra and derivative，it might be a little big hard to understand the gradient and the convergence procedure of a regression, but if you have enough Maths knowledge, it is easy to understand.

Logistic Regression can be considered as a varient of Linear Regression for a logistic result. So a more natural learning path of regression might be to learn Linear Regression first. The main difference of Logistic Regression from Linear Regression is, the word "Logistic" indicating that, for each feature, the predict cares about only a probability value in range [0, 1], while Linear Regression's prediction result can be an arbitrary number value. Logistic Regression uses a logistic sigmoid function to convert an arbitrary value to a probablity value in range [0, 1] which can then be mapped to two or more discrete classes.

Andrew Ng has a very popular [machine learning course on coursera](https://www.coursera.org/learn/machine-learning), the first several chapters of which step-by-step introduce from linear regression to logistic regression with all the Maths knowledge behind them clearly, which is highly recommended if you meet any troubles in understanding this chapter.

Logistic Regression is one of the most important machine learning algorithms. It is the core component of neural networks and deep learning coz each node of a hidden or output layer in neural networks doing exactly a Logistic Regression. So it is worthy to pay more attention in it and to fully understand it.

## Demo Code

[logRegres.py](logRegres.py) - Revised version of the original logRegres demo

[logRegresSklearn.py](logRegresSklearn.py) - A scikit-learn version of logRegres implementation

[logRegresPlotter.py](logRegresPlotter.py) - The scripts for plotting figures in this chapter
