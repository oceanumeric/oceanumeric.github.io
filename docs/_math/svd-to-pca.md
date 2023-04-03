---
title: From SVD to PCA
subtitle: The applications of SVD are manifold. In this post, we will focus on the application of SVD to PCA, which is a great tool for dimensionality reduction.
layout: math_page_template
date: 2023-04-03 
keywords: linear-algebra, svd, pca, dimensionality-reduction
published: true
tags: probability algorithm data-science machine-learning numerical-linear-algebra
---

Principal Component Analysis (PCA) is a dimensionality reduction technique that is widely used in data science. In this post, we will focus on the application of SVD to PCA, which is a great tool for dimensionality reduction. 

First, we will learn how to find eigenvalues and eigenvectors of a matrix. Then, we
will learn how to use SVD to find principal components.

- [Eigenvalues and eigenvectors](#eigenvalues-and-eigenvectors)
- [Basic QR iteration](#basic-qr-iteration)

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




























{% endkatexmm %}