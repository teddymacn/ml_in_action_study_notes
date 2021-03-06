# Linear Regression

Chapter 8 talks about Linear Regression. Linear Regression is perhaps one of the most well known and well understood algorithms in linear algebra, statistics and machine learning.

Many people like me heard about Linear Regression the first time in the linear algebra lectures in college. To recall a little bit, in linear algebra:
- Given y is a N&#00215;1 matrix and equals to a linear combination of a N&#00215;(M+1) matrix X;
- The first column of X is all constant 1, represents the constants part of the linear combination;
- w is a (M+1)&#00215;1 matrix, which is the weights of the linear combination
- And given N >= M， which is necessary, otherwise X becomes sigular and X&#00175;&#00185; not exists
- So we have: y = Xw
- And then we get: w = (X&#07488;X)&#00175;&#00185;X&#07488;y

With the formula above, and since we already have the training data X and y, as long as the numbers of features and training examples N and M are not too large values, we are able to compute out the weights w by linear algebra operations directly. 

The linear algebra way is perfect for small training datasets. But when we have much more features or training examples, the computation cost grows up exponentially and might become inefficient in computation. Then we need a more efficient algorithm to fit the weights, and the most popular way is called Gradient Descent. This chapter doesn't talk too much about Gradient Descent for fitting Linear Regression, which is a shame. But it is already discussed and demonstrated in [Logistic Regression](../ch05/README.md) in chapter 5, you can use the exact same algorithm in Linear Regression for fitting large training dataset.

8.4.1 talks about a popular variant of Linear Regression which is called Ridge Regression. Ridge Regression was originally developed to deal with the problems of having more features than training examples. But it can also be used to add bias into our estimations to decrease unimportant parameters. This decreasing is known as shrinkage in statistics. The formula of w in Ridge Regression becomes:

w = (X&#07488;X + &#00955;I)&#00175;&#00185;X&#07488;y

Here the note I after &#00955; represents an identity matrix and &#00955; is a constant scalar value as a penalty. In practice, before fitting the weights, we need to specify the value of &#00955; first. So usually we will run the fitting procedure multiple times with different values for &#00955;, and finally we can get a (&#00955;,w) combination which could give us the least error rate from all the &#00955;s.

Furthermore, in reality, very rare cases we have a linear dataset directly, when it is for a non-linear dataset, we might want to fit the weights of higher order combination of x to y, something like y = w&#008320; + w&#008321;x&#008321; + w&#008322;x&#008322;&#00178; + ..., this is called [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression). But we could treat the mapping of [x&#008321; x&#008322; ...] to [x&#008321; x&#008322;&#00178; ...] as a preprocessing of a Linear Regression. So that we could use Linear Regression to implement Polynomial Regression. It is again one more shame that this book doesn't talk about Polynomial Regression at all, which is actually more often to happen in real life datasets than simple Linear Regression.

I'd like to recommend Andrew Ng's popular [machine learning course on coursera](https://www.coursera.org/learn/machine-learning) again, the first several chapters of which step-by-step introduce Linear Regression and how to use Gradient Descent to fit the weights, including the case of having the penalty &#00955; and the cases of polynomial regression, etc.

## Demo Code

[regression.py](regression.py) - Revised version of the original regression demo
