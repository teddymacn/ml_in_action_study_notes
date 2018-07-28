# Tree-based Regression

Chapter 9 talks about another popular category of regression, tree-based regression.

Firstly, tree-based algorithms, such as Decision Trees, can not only be used to do classifications, but also to do regressions directly. The only difference is, in classification, each prediction output is a class index; while in regression, each prediction output is a float target value.

But more popularly, tree-based algorithms are used together with Linear Regression or Polynomial Regression. A typical case is what discussed in chapter 9.5 Figure 9.4, when a non-linear dataset can be splitted into several partitions and each partition is linear. In this kind of scenarios, we could use a decision tree to split the dataset into several partitions first, and then to use Linear Regression to easily fit for each partition.

## Demo Code

[regTrees.py](regTrees.py) - Revised version of the original tree based regression demo

[treeExplore.py](treeExplore.py) - The scripts to plot the GUI demo in chapter 9.7
