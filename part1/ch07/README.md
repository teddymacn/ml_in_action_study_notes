# AdaBoost

Chapter 7 of this book talks about a new category of machine learning algorithms, which are called ensemble methods.

The word "ensemble" means to combine the predictions of several estimators rather than to depend on a single one.

Ensemble methods can be divided into two groups:

- Sequential ensemble - the base estimators are generated sequentially (e.g. AdaBoost)
- Parallel ensemble - the base estimators are generated in parallel (e.g. Bagging)

This chapter mainly discusses the most widely used sequential ensemble method - AdaBoost. It is considered by some people to be the best-supervised learning algorithm.

The word "AdaBoost" means adaptive boosting, the idea of AdaBoost is to fit a sequence of weak learners (such as small decision trees) on repeatedly modified versions of the data, usually adjust the weights of test data in each succeeding iteration based on the result of previous iteration to produce the final prediction. Each iteration is called a boosting iteration. Coz the picked learner is weak, it is easy to understand and cheap in computaton.

## Demo Code

[adaboost.py](adaboost.py) - Revised version of the original adaboost demo

[adaboostSklearn.py](adaboostSklearn.py) - A scikit-learn version of adaboost implementation