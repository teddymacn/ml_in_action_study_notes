# Principal Component Analysis (PCA)

Since Chapter 13, this book talks about several dimensionality reduction techniques for supporting machine learning, including Principal Component Analysis (PCA) and Singular
Value Decomposition (SVD).

Dimensionality reduction is quite important in machine learning especially when we have hundreds or sometimes thousands of features. It helps in:
- Making the dataset easier to use
- Reducing computational cost of many algorithms
- Removing noise
- Making the results easier to understand

Dimensionality reduction is useful, but please always realize that it actually drops some information from the original dataset. So only do dimensionality reduction when you have to.

Chapter 13 talks about PCA which is one of the most widely used dimensionality reduction techniques. The idea of PCA bases on the concept of eigenvalues and eigenvectors in linear algebra. The detailed steps are:

1. Calculate the mean of the dataset;
2. Remove the mean from each value of the matrix;
3. Compute the covariance matrix of the matrix after step 2;
4. Find the eigenvalues and eigenvectors of the covariance matrix;
5. Sort the eigenvalues from largest to smallest;
6. Take the top N eigenvectors;
7. Transform the data into the new space created by the top N eigenvectors;
8. Add back the mean of step 1 to each value of the new matrix;

Why PCA works is because, the eigenvalues actually represents the importance of the related data in the matrix, so if we remove related data of some small eigenvalues, we get a lower dimensional matrix, but we still have most of the information of the dataset.

## Demo Code

[pca.py](pca.py) - Revised version of the original pca demo
