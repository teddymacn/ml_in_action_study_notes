# Logistic Regression

In chapter 5, we encountered the first optimization algorithm, specifically the first regression algorithm: Logistic Regression.

Frankly speaking, comparing to the previous chapters, this chapter discusses too less about the related Maths behind regression, so if you have limited Maths knowledge about linear algebra and derivative，it might be a liile big hard to understand the gradient and the convergence procedure of a regression.

And actually, Logistic Regression is a varient of Linear Regression, so a more natural learning path of regression might be to learn Linear Regression first. The main extension of Logistic Regression from Linear Regression is, the word "Logistic" indicating that, for each feature, the predict cares about only a true or false result, while Linear Regression's predict result can be an arbitrary number value. Logistic Regression uses a sigmoid function to convert an arbitrary value to a value in range [0, 1].

Andrew Ng has a very popular [machine learning course on coursera](https://www.coursera.org/learn/machine-learning), the first several chapters of which step-by-step introduce from linear regression to logistic regression with all the Maths knowledge behind them clearly, which is highly recommended if you meet any troubles in understanding this chapter.

In the demo of this chapter, the testing dataset only has two features, and it only trys to fit a simple linear equation of the two features. In the corresponding 2-D linear regression, the optimal class separator will just be a straight line. So it will not be a perfect estimation of a function for the real dataset. That's part of the reason why even the best final error rate is around 30%. But as a demo for the Logistic Regression idea, it should be enough. But we should realize that, in reality, if we want to get a more accurate estimation function, we might try to fit a more complex equation of the features.

## Demo Code

[logRegres.py](logRegres.py) - Revised version of the original logRegres demo

[logRegresSklearn.py](logRegresSklearn.py) - A scikit-learn version of logRegres implementation

[logRegresPlotter.py](logRegresPlotter.py) - The scripts for plotting figures in this chapter