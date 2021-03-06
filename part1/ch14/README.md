# Singular Value Decomposition (SVD)

Chapter 14 talks about Singular Value Decomposition (SVD), which is another popular dimensionality reduction technique in machine learning.

SVD is also a linear algebra concept, which is a popular way in linear algebra for data compression based on rank reduction. The idea of SVD bases on the following formula. For any m&#00215;n matrix D, we could always convert it to the product of 3 matrix U, &#00931; and V&#07488;, where U is m&#00215;m, &#00931; is m&#00215;n and V&#07488; is n&#00215;n.

D = U&#00931;V&#07488;

This does the so-called decomposition, which creates the matrix &#00931; containing only diagonal elements, all the other elements are 0, and the diagonal elements are sorted from largest to smallest. These diagonal elements are the singular values of our original data set D. Looks quite similar to what we have in PCA for eigenvalues, right? Yes, actually, the singular values are the square root of the eigenvalues
of the product of matrix DD&#07488;.

As long as we convert the original dataset to the formula above, what SVD will do is, it will pick only the top k singular values and drop the other ones. Thus the dimension of &#00931; becomes k&#00215;k, U becomes m&#00215;k and V&#07488; becomes k&#00215;n. After this procedure, you actually have removed some unimportant information related to those dropped small singular values from the original matrix, the rank of the original matrix is reduced from n to k although the dimension of the new matrix D is still m&#00215;n.

But because the rank of the matrix is reduced to k, when storing this matrix, you only need to store the full elements of k vectors of the matrix, and the other vectors of the matrix should all be linear combination of the k vectors. So for all the other vectors, you only need to store their weights of the linear combination, which reduces a lot of space for storing. This is the basic idea of how SVD works for data compression.

## Demo Code

[svd.py](svd.py) - Revised version of the original svd demo
