---
title: From SVD to PCA
subtitle: The applications of Singular Value Decomposition (SVD) are manifold. In this post, we will focus on the application of SVD to PCA, which is a great tool for dimensionality reduction.
layout: math_page_template
date: 2023-04-03 
keywords: linear-algebra, svd, pca, dimensionality-reduction
published: true
tags: probability algorithm data-science machine-learning numerical-linear-algebra
---

Principal Component Analysis (PCA) is a dimensionality reduction technique that is widely used in data science. In this post, we will focus on the application of Singular Value Decomposition (SVD) to PCA, which is a great tool for dimensionality reduction. 

First, we will learn how to find eigenvalues and eigenvectors of a matrix. Then, we
will learn how to use SVD to find principal components.

- [Eigenvalues and eigenvectors](#eigenvalues-and-eigenvectors)
- [Basic QR iteration](#basic-qr-iteration)
- [Improving the convergence of QR iteration](#improving-the-convergence-of-qr-iteration)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)

## Eigenvalues and eigenvectors

{% katexmm %}


One of the top 10 algorithms of the 20th century is the QR iteration for computing
eigenvalues and eigenvectors of a matrix. it transformed a problem that seems
incredibly difficult into something that can be solved numerically efficiently and
reliably. 

Before we dive into the QR iteration, let's dicuss _Abel-Ruffini Theorem_ a little bit.
This theorem states that there is no general algebraic solution to polynomial
equations of degree greater than 4. This theorem is the reason why we can't solve
polynomial equations of degree greater than 4, which means we cannot compute
eigenvalues and eigenvectors of a matrix of order greater than 4 either. That's why 
we need to use the QR iteration.


Let's denote by $A = X \Lambda X^{-1}$ the eigenvalue decomposition of a matrix $A$.
In this post, many of our algorithms will __normalize vectors__ by dividing them their
norm. For example, if $x$ is a vector, then $x / \|x\|$ is the normalized vector. 


Suppose that $A$ is a _symmetric_ matrix of order $n$. Then $A$ as an eigenvalue decomposition
$A = X \Lambda X^{-1}$. One of the nice things about this expression is that
it provides a very simple expression for $A^k$:

$$
A^k = X \Lambda^k X^{-1} = X \Lambda X^{-1} X \Lambda X^{-1} \cdots X \Lambda X^{-1} \tag{1}
$$

This equation can be expanded as follows:


$$
\begin{aligned}
A^k & = \sum_i \lambda_i^k x_i y_i^H \\
    & = \lambda_1^k x_1 y_1^H + \lambda_2^k x_2 y_2^H + \cdots + \lambda_n^k x_n y_n^H \tag{2}
\end{aligned} 
$$


Notice that even if $A$ is real, it might have complex eigenvalues and eigenvectors,
which is why we are using the conjugate transpose instead of just the transpose. 

<div class='figure'>
    <img src="/math/images/power-iteration1.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of the eigenvalue decomposition of a matrix $A$.
    </div>
</div>

_Remark_: one needs to ponder a little bit to understand why the above equation will
work by looking at the figure 1, each term in the sum of the equation (2) is a matrix with dimension $n \times n$.



Let's now imagine that the eigenvalues satisfy the following condition:

$$
|\lambda_1| \geq |\lambda_2| \geq \cdots \geq |\lambda_n| \tag{3}
$$

Then, even for moderate values of $k$, we expected $\lambda_1^k$ to be much larger than
others, which will dominate the sum in equation (2). This means that the dominant
eigenvector of $A$ is the eigenvector corresponding to the largest eigenvalue:

$$
A^k \approx \lambda_1^k x_1 y_1^H \tag{4}
$$

Now, let's multiply both sides of equation (4) by a random vector $v$:

$$
A^k v \approx \lambda_1^k x_1 y_1^H v \tag{5}
$$


Since the dimension of the vector $v$ is $n \times 1$, which is same with 
the dimension of $x_1$, we can write the above equation as follows:

$$
A^k v \approx (\lambda_1^k y_1^H v) x_1  \tag{6}
$$

The power iteration algoritm is based on the above equation. The algorithm is as follows:

1. Initialize $v$ to a random vector of dimension $n \times 1$.
2. Repeat the following steps until convergence:
    1. Compute $v \leftarrow A v$.
    2. Normalize $v$ by dividing it by its norm.
    3. Compute $\lambda \leftarrow v^H A v$.

The algorithm will converge to the dominant eigenvector of $A$ and the corresponding
eigenvalue. Notice that we never need to compute the matrix $A^k$ explicitly as the
power iteration algorithm only needs to compute $A^k v$ for a random vector $v$.

```python
import numpy as np


def power_iteration(A:np.ndarray, num_simulations:int):
    """Returns the dominant eigenpair of A.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    num_simulations : int
        Number of power iterations to perform.

    Returns
    -------
    v : ndarray
        Dominant eigenvector.
    lambda : float
        Dominant eigenvalue.
    """
    n, m = A.shape
    if n != m:
        raise ValueError('A must be a square matrix.')
    
    # generate a random vector
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    for _ in range(num_simulations):
        Av = A @ v
        v = Av / np.linalg.norm(Av)

    lambda_top = v.T @ A @ v
    return v, lambda_top
```

Although the power iteration algorithm above is great but it is somewhat limited.
First, it cannot be used to compute all eigenvalues and eigenvectors of a matrix.
Second, if we already have a good approximation of one of the eigenvalues, there
is no way to use this information to accelerate the convergence of the algorithm.


Now, we will introduce a method called _inverse iteration_ that can be used
to find an approximate eigenvector when the _an approximate eigenvalue is known_.

Assume that we have an approximation $\mu$ as one of the eigenvalues of $A$, say
$\lambda_i$ (which means that $\mu$ is close to $\lambda_i$). Then, consider
the matrix

$$
A - \mu I ,
$$

which should have very small eigenvalues $\lambda_i - \mu$ and its inverse is

$$
(A - \mu I)^{-1},
$$

must have one very big eigenvalue $1/( \lambda_i - \mu)$. The inverse iteration
algorithm is as follows:

1. Initialize $v$ to a random vector of dimension $n \times 1$.
2. Repeat the following steps until convergence:
    1. Compute $v \leftarrow (A - \mu I)^{-1} v$.
    2. Normalize $v$ by dividing it by its norm.
    3. Compute $\lambda \leftarrow v^H A v$.

The above algorithm should allow us to find an approximate eigenvector of $A$
very quickly since we are using extra information about the eigenvalue. 

Actuall we could even speed up the convergence of the algorithm by using Rayleigh
quotient iteration. The idea is to use the following equation to compute the

$$
\lambda = \frac{v^H A v}{v^H v} \tag{7}
$$

The algorithm is as follows:

1. Initialize $v$ to a random vector of dimension $n \times 1$.
2. Repeat the following steps until convergence:
    1. Compute $v \leftarrow (A - \mu I)^{-1} v$.
    2. Normalize $v$ by dividing it by its norm.
    3. Compute $\mu \leftarrow v^H A v$.
    4. Compute $\lambda \leftarrow v^H A v$.



## Basic QR iteration

The QR iteration algorithm is a very powerful algorithm that can be used to compute
__all__ eigenvalues and eigenvectors of a matrix instead of just one.


From now on, we no longer assume that the matrix $A$ is symmetric. We will only
assume that the matrix $A$ is _diagnoalizable_ with eigenvalues such that
$|\lambda_1| \geq |\lambda_2| \geq \cdots \geq |\lambda_n|$.

We begin with an algorithm called orthogonal iteration, which allows us 
to revover more than one eigenvalues at once. Since eigenvalues are 
just basis vectors of the matrix $A$, we can construct orthogonal basis
during our power iteration. 

Let's walk through for a special case, which assumes that the matrix $A$ has
$r = 2$ distinct eigenvalues. The algorithm is as follows:

1. Choose two random orthogonal vectors $v_1$ and $v_2$.
2. Repeat the following steps until convergence:
    1. Compute $v_1 \leftarrow A v_1$, $v_2 \leftarrow A v_2$.
    2. Project $v_2$ onto the orthogonal complement of $v_1$.
    3. Normalize $v_1$ and $v_2$. 

This means $v_1$ and $v_2$ span a subspace of dimension $r = 2$ that is
orthogonal to each other. The algorithm will converge to two eigenvectors
of $A$ that are orthogonal to each other.

To find the eigenvalues, we need to compuate  the following values:

$$
Q_k^T A Q_k, \quad \quad Q_k = [v_1, v_2] \tag{8}
$$


<div class='figure'>
    <img src="/math/images/qr-iteration1.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The illustration of QR iteration.
    </div>
</div>

The reason that we call this method _QR iteration_ is that $Q_kAQ_k$ converges
to a matrix that is upper-triangular, with $\lambda_1$ and $\lambda_2$ on the
diagonal as it is shown in Figure 2.

To see why it is upper-triangular, notice that 

$$
v_2^T A v_1 \approx v_2^T A x_1 = \lambda_1 v_2^T x_1 \approx \lambda_1 v_2^T v_1 = 0,
$$

and so the lower-left entry converges to 0. 

The key idea of the QR iteration algorithm is to use the QR decomposition to
construct the basis we need in equation (8). We want to have a series of
vectors that are orthogonal to each other and span the same subspace as
eigenvectors of $A$. The QR decomposition allows us to construct such a
basis.

Recall that the QR decomposition of a matrix $A$ of $m \times n$ is given by

$$
A = QR, \quad \quad Q^T Q = I, \quad \quad R \in \mathbb{R}^{n \times n} \tag{9}
$$

where $Q$ is an orthogonal matrix and $R$ is an upper-triangular matrix. If you
want to refresh your memory about the QR decomposition, you can check out
[my another post](https://oceanumeric.github.io/math/2023/04/QR-factorization){:target="_blank"}.


```python
def qr_iteration(A:np.ndarray, num_simulations:int):
    """Returns all eigenpairs of A.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    num_simulations : int
        Number of QR iterations to perform.

    Returns
    -------
    v : ndarray
        all eigenvectors.
    lambda : float
        all eigenvalues.
    """
 
    for _ in range(num_simulations):
        Q, R = np.linalg.qr(A)
        A = R @ Q

    v = Q
    lambda_all = A.diagonal()
    return v, lambda_all
```


_Remark_: when the matrix $A$ is symmetric, there are more efficient ways to find the eigenvalues and eigenvectors, such as Jacobi method and Householder method. We will not cover these methods in this post.


## Improving the convergence of QR iteration

The QR iteration algorithm is very powerful, but it is not very efficient. The reason is that the QR decomposition is very expensive to compute. In fact, the QR decomposition is $O(n^3)$, which is much more expensive than the power iteration algorithm. There are two ways to improve the convergence of the QR iteration algorithm:

- Transform the matrix $A$ into upper Hessenberg form.
- Use the shift-invert method.

The first variant is called the _Hessenberg QR iteration_ and the second variant is called the _shifted QR iteration_. The first one is very easy to implement, we just need to transform the matrix $A$ into upper Hessenberg form and then perform the QR iteration. The second one is a little bit more complicated, but it is still very easy to implement.

```python
import scipy as sp


def qr_iteration_with_hessenberg(A:np.ndarray, num_simulations:int):


    H = sp.linalg.hessenberg(A)
    for _ in range(num_simulations):
        Q, R = np.linalg.qr(H)
        H = R @ Q

    v = Q
    lambda_all = H.diagonal()
    return v, lambda_all
```

The Hessenberg QR iteration algorithm is much more efficient than the basic QR iteration algorithm. The reason is that the Hessenberg QR iteration algorithm only requires $O(n^2)$ operations to compute the QR decomposition. The basic QR iteration algorithm requires $O(n^3)$ operations to compute the QR decomposition. However, the Hessenberg QR iteration algorithm is still not as efficient as it converges much slower
than shifted QR iteration algorithm.

The shifted QR iteration algorithm combines the QR iteration algorithm with the shift-invert method, which uses equation (7) to compute the eigenvalues. When we talk about _inverse iteration_, we state that if we know an approximation $u$ of an eigenvector, then we can compute the eigenvalue by using the following matrix

$$
(A- uI)^{-1} \tag{10}
$$

as its eigenvalue must be very big. The shifted QR iteration algorithm uses the same idea. 

Notice that if $A_k$ - which converging to upper-triangular - had the following form

$$
A_k = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & \cdots & a_{1n} \\
a_{21} & a_{22} & a_{23} & \cdots & a_{2n} \\
 & a_{32} & a_{33} & \cdots & a_{3n} \\
    &  & \ddots & \ddots & \vdots \\
    &  &  & 0 & a_{nn} \\
\end{bmatrix} = \begin{bmatrix}
B_{11} & U \\
0^T & \lambda
\end{bmatrix}
$$

Then $\lambda$ is the eigenvalue of $A_k$ and the remaining matrix $B_{11}$ has the same eigenvalues as $A_k[:n-1, :n-1]$. The shifted QR iteration algorithm uses this idea to compute the eigenvalues.

Here is the implementation of the shifted QR iteration algorithm.

1. Set $A = Q_H^T A Q_H$ where $Q_H$ is the orthogonal matrix that transforms $A$ into upper Hessenberg form.
2. Repeat the following steps until convergence:
    1. Compute a shift $u$ by using equation (11).
    2. Compute $Q_kR_k \leftarrow (A_{k-1} - uI)$.
    3. Set $A_k = R_kQ_k + u_k I$.

We are not using Rayleigh quotient to compute the shift $u$ because it is very expensive to compute the inverse of a matrix. Instead, we use the 
Wilkinson's shift, which is given by

$$
B = \begin{bmatrix}
a & b \\
b & c
\end{bmatrix} \quad \quad u = c - \frac{\mathrm{sign}(\delta)b^2}{|\delta| + \sqrt{\delta^2 + b^2}} \tag{11}
$$

where $\delta = (a - c)/2$. 

Here is the implementation of the shifted QR iteration algorithm.

```python
def wilkinson_shift(a, b, c):
    # Calculate Wilkinson's shift for symmetric matrices: 
    delta = (a-c)/2
    shift = c - np.sign(delta)*b**2/(np.abs(delta) + np.sqrt(delta**2+b**2))
    return shift


def qr_with_shift(A:np.ndarray, num_iterations:int):
    
    n, m = A.shape
    eigen_values = []
    if n != m:
        raise ValueError('A must be a symmetric matrix.')
    
    I = np.eye(n)
    
    H = sp.linalg.hessenberg(A)
    for _ in range(num_iterations):
        u = wilkinson_shift(H[n-2, n-2], H[n-1, n-1], H[n-2, n-1])
        Q, R = np.linalg.qr(H - u*I)
        H = R @ Q + u*I
    
    v = Q
    lambda_all = H.diagonal()
    return v, lambda_all

```

## Singular Value Decomposition (SVD)

For a matrix $A$ of  $m \times n$,  the SVD is given by

$$
A = U \Sigma V^T \tag{12}
$$

where $U^T U = I_m$ and $V^TV = I_n$. How could we find the SVD of a matrix? 

Calculating the SVD of a matrix is very easy. We just need to find the eigenvalues and eigenvectors of $A^TA$ and $AA^T$ because

$$
A^TA = V \Sigma^2 V^T \quad \quad AA^T = U \Sigma^2 U^T \tag{13}
$$

In particular, the eigenvectors of $A^TA$ are the columns of $V$ and the eigenvectors of $AA^T$ are the columns of $U$. The eigenvalues of $A^TA$ are the squares of the singular values of $A$ and the eigenvalues of $AA^T$ are the squares of the singular values of $A$.


## Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a very powerful technique that is used to reduce the dimensionality of a dataset. The idea is that we can find a new basis that is orthogonal to each other and that is a linear combination of the original basis. The new basis is called the _principal components_ and the linear combination is called the _principal components scores_. The principal components are the eigenvectors of the covariance matrix of the dataset. The principal components scores are the eigenvalues of the covariance matrix of the dataset.


Suppose we have a dataset $X$ of $n$ samples and $m$ features. The covariance matrix of $X$ is given by

$$
\Sigma = \frac{1}{n} X^T X \tag{14}
$$

The covariance matrix $\Sigma$ is a symmetric matrix. The eigenvectors of $\Sigma$ are the principal components of $X$ and the eigenvalues of $\Sigma$ are the principal components scores of $X$.

When the matrix $A$ is symmetric, the eigenvectors of $A$ are orthogonal to each other. This means we can have

$$
A = U \Lambda U^T \tag{15}
$$

<div class='figure'>
    <img src="/math/images/linear-system-decomposition.png"
         alt="floating number illustrated"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of the SVD transformation.
    </div>
</div>


What PCA does is to project the data onto the principal components, which are the eigenvectors corresponding to the largest eigenvalues.

The PCA method makes the fundamental assumption that the data is centered. This means that the mean of each feature is zero. If the data is not centered, then we need to subtract the mean of each feature from the data before applying PCA. It also makes the fundamental assumption that the data is normalized. This means that the variance of each feature is one. If the data is not normalized, then we need to divide each feature by its standard deviation before applying PCA.

Then, we will have a linear transformation which makes the data have variables that is as much uncorrelated as possible. 
















{% endkatexmm %}