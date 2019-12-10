title: Singular Value Decomposition
slug: svd
category: 
date: 2019-11-22
modified: 2019-11-22
tags: machine learning, linear algebra, natural language processing


## Introduction

<!-- PELICAN_BEGIN_SUMMARY -->

Some quick notes about Singular Value Decomposition (SVD) to develop an intuition that will help solve problems related to collaborative filtering, natural language processing (NLP), dimensionality reduction, image compression, denoising data etc.

<!-- PELICAN_END_SUMMARY -->

Let's imagine we have a matrix $\textbf{A}$ of size $m$ by $n$ where $m$  is the number of rows and  $n$ is the number of columns. If we were building a recommender system, we can think of each row representing a user, each column representing an item, and each element in the matrix indicating whether the user has interacted with the item. In NLP, we can think of each row representing a document, and each column representing a term.

The goal of singular value decomposition (SVD) is to take this matrix $A$ and represent it as the product of three matricies:

$$\textbf{A} = \textbf{U}\mathbb{\Sigma}
\textbf{V}^T$$

The dimensions of $U$ are $m$ by $r$. This matrix stores the left  singular vectors. The dimensions of $\mathbb{\Sigma}$ is $r$ by $r$. This is a diagonal matrix, and contains singular values. The dimensions of $V$ are $n$ by $r$. Before we get into why this decomposition is useful, let's talk a little bit about the properties of the singular value decomposition of a matrix. Namely:

It is always possible to decompose a real matrix $\textbf{A}$ (meaning all values are real numbers) into $\textbf{A} = \textbf{U}\mathbb{\Sigma}
\textbf{V}^T$. In addition, there is only one unique singular value decomposition for each  

So why do people use SVD? I think the most interesting way  people use SVD is to create embeddings, which we will talk about next (and will help us understand the three matricies of SVD better).

### Using SVD for embeddings

So let's take the example inspired from this [video](https://www.youtube.com/watch?v=P5mlg91as1c&t=236s), where we have a matrix $\textbf{A}$ that has the ratings users give to  movies. Let's say we want to figure out which movies are similar and which users are similar.

$$\textbf{A} = \begin{bmatrix}
  1 & 1 & 4 \\
   5 & 5 & 1 \\
   2 & 1 & 5
\end{bmatrix}$$

The rows of our matrix represent users, the columns of our matrix represent movies, and the values represent the rating a user gives to a movie. Since Disney+ just came out, let's imagine that our three movies are *Star Wars*, *Avengers*, and *Lady and the Tramp*. Let's take a look at the ratings that are users gave our three movies. Right away, you can tell that user 1 and user 3 are similar because they both  disliked *Star Wars* and *Avengers* and liked *Lady and the Tramp*. User 2 is the opposite because she likes both *Star Wars* and *Avengers* and dislikes *Lady and the Tramp*.

Let's now take a look at the  SVD  for this matrix. We can use the `numpy` function `svd` to solve it quickly:

    import numpy as np

    A = np.array([[1,1,4], [5,5,1], [2,1,5]]) 
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # convert S to diagonal matrix
    S = np.diag(S)

The results that we get after running this code are as follows:

$$\textbf{U} = \begin{bmatrix}
  -0.40865502 & 0.47145889  & -0.78149062 \\
   -0.73017597 & -0.68260113 & -0.02997898 \\
   -0.54758024 & 0.55837461 & 0.62319633
\end{bmatrix}$$

$$\mathbb{\Sigma} = \begin{bmatrix}
  8.59266627 & 0 & 0 \\
  0 & 4.99702707 & 0 \\
  0 & 0 & 0.44250069
\end{bmatrix}$$


$$\textbf{V}^T = \begin{bmatrix}
  -0.59989475 & -0.53616828 & -0.5938433 \\
   -0.36517664 & -0.476918 & 0.79949687 \\
   0.71187942 & -0.69647167 & -0.09030447
\end{bmatrix}$$

We can think of the rows of the matrix $U$ as embeddings for our three users. Similarly, we can think of the three columns as the embeddings for our three movies. You can think of embeddings as summaries of what we know about things (in this case, what we know about users and movies).

So how can we use these embeddings? The most straightforward thing we can do is use them to find the similarity between users or movies. So recall from above that we posited that user 1 and user 3 have similar preferences.

The first value of user 1's embedding is -0.4087. Similarly, the first value of user 2's embedding is -0.7302 and the first value of user 3's embedding is -0.5476. 

Notice that the absolute value of the difference between user 1 and user 2's values is 0.1389 whereas the difference between user 1 and user 3's values is 0.3215. So that matches our intuition, that user 1 and user 2 will  have similar embeddings because they have similar preferences. Cool!

Why did we pick the first value of the embedding vector to compare instead of the second or third? The values are sorted by importance. You can check the values importance by looking at the diagonal values of $\Sigma$:

$$[8.5927, 4.9970, 0.4425]$$

What this tells you is that the first value of the embeddings is roughly twice as "important" as the second value, and twenty times as "important" as the third value.

Notice that we could have compared the embeddings for movies by using the matrix $V$ instead of $U$.

So what people usually do to create embeddings for users, items, words, documents, etc. is to take the top $k$ values from the vectors created using SVD. Some people refer to this strategy as Latent Semantic Analysis, or LSA.

How do you know what the value of $k$ is? A lot of people use trial and error, or by checking the [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

Once you create the embeddings, how do you tell how similar embeddings are? People use cosine similarity, manhattan distance ($L1$), or euclidian distance ($L2$). In the example above, I used manhattan distance. Again, there's no default best metric to use and in practice people use trial and error.

Some of the drawbacks of this method (as mentioned [here](https://github.com/stanfordnlp/cs224n-winter17-notes/blob/master/notes1.pdf)): 

- Dimensions of the matrix can change often (adding new users or movies), and each time the dimensions change, SVD must be performed again. 
- The matrix can be very high dimensional
- Computational cost to perform SVD for a $m$ by $n$ matrix is $O(mn$)

For these reasons, iterative methods tend to be better, the simplest of these being [word2vec](/word2vec).


### Resources
1)  [Great video on SVD that explains the math](https://www.youtube.com/watch?v=P5mlg91as1c&t=236s)

2) [Using SVD to coompress an image example](http://andrew.gibiansky.com/blog/mathematics/cool-linear-algebra-singular-value-decomposition/)

3) [Notes from Stanford NLP course](https://github.com/stanfordnlp/cs224n-winter17-notes/blob/master/notes1.pdf)