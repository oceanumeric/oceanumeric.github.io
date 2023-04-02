---
title: Solving Linear Systems
subtitle: Linear systems of equations are the bread and butter of numerical linear algebra. Solving them is at the core of many machine learning algorithms and engineering applications.
layout: math_page_template
date: 2023-04-01
keywords: linear systems, linear algebra, numerical linear algebra, machine learning, data science, python
published: true
tags: algorithm data-science machine-learning numerical-linear-algebra python
---


Although many packages and software libraries exist for solving linear systems, it is important to understand the underlying algorithms and how they work. This will help us to develop a mental model of matrix operations and how they relate to the underlying data. In this section, we will cover the following topics:

- [LU factorization](#lu-factorization)
- [LU factorization with pivoting](#lu-factorization-with-pivoting)


## LU factorization

{% katexmm %}

The LU factorization is a method for decomposing a matrix into a lower triangular matrix and an upper triangular matrix. The LU factorization is a very important algorithm in numerical linear algebra. It is used in many other algorithms, such as the QR factorization, the Cholesky factorization, and the singular value decomposition. The LU factorization is also used in many machine learning algorithms, such as the linear regression algorithm and the principal component analysis algorithm.

Now, for a linear system $Ax = b$, we can write $A = LU$, where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix as it is illustrated in the following figure { % cite darve2021numerical % }.


<div class='figure'>
    <img src="/math/images/lu-illustration.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of the LU factorization. The figure is taken from the book by Darve and Wootters (2021). 
    </div>
</div>

To solve, first we compute $x_1$:

$$x_1 = \frac{b_1}{l_{11}}$$

After solving for $x_1$, we can obtain $x_2$: 

$$x_2 = \frac{1}{l_{22}} (b_2 - l_{21} x_1)$$

From there, $x_3, x_4, \cdots$, can be computed up to $x_n$. Now, let's
implement this in JAX. 


```python
import jax.numpy as jnp
from jax import jit, random, lax


def row_wise_loop(L: jnp.ndarray, b: jnp.ndarray):
    """
    Forward substitution with row-wise loop based on the above formula
    L: m by n matrix
    b: m by 1 matrix
    """
    m, n = L.shape
    x = jnp.zeros_like(b, dtype=jnp.float32)
    # calculate the first one x_1 = b_1/l_{11}
    x = x.at[0, 0].set(b[0, 0]/L[0, 0])  # jax array is immutable 
    # # loop over rows, starting from the second one
    for i in range(1, m):
        # loop over column elements  
        # initialize the dot product
        z = 0.0
        for j in range(i):
            # calculate the dot product
            z += L[i, j] * x[j]

        solution = (b[i, 0] - z) / L[i, i]  # array
        # update x_i
        x = x.at[i, 0].set(solution[0])  # set value 
    
    return x
```

__Solving general systems and the LU factorization__. To solve $Ax = b$, our
strategy will be to _factor_ $A$ as $A = LU$ for a lower-triangular matrix
$L$ and an upper-triangular matrix $U$. Let $A$ be a __square__ $n \times n$ matrix. An __LU factorization__ is a 
factorization of the form $A=LU$, where $L$ is lower-triangular and 
$U$ is upper-triangular. 

If we have $PA = LU$ and $Ax =b$, then 

$$PAx = Pb  \ \ \to LUx = Pb  \ \ \to Lz = Pb \ \ (z = Ux) $$


Since the diagonal entries of $L$ are all 1 , the triangular system 
$L \mathbf{z}=\mathbf{b}$ has the form

$$
\left[\begin{array}{ccccc}
1 & 0 & 0 & \cdots & 0 \\
l_{21} & 1 & 0 & \cdots & 0 \\
l_{31} & l_{32} & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
l_{n 1} & l_{n 2} & l_{n 3} & \cdots & 1
\end{array}\right]\left[\begin{array}{c}
z_1 \\
z_2 \\
z_3 \\
\vdots \\
z_n
\end{array}\right]=\left[\begin{array}{c}
b_1 \\
b_2 \\
b_3 \\
\vdots \\
b_n
\end{array}\right] .
$$


This gives the forward substitution as th matrix multiplication yields the equations
$$
\begin{array}{rlrl}
z_1 & =b_1, & z_1 & =b_1, \\
l_{21} z_1+z_2 & =b_2, & z_2 & =b_2-l_{21} z_1, \\
\vdots & \vdots & \vdots \\
\sum_{j=1}^{k-1} l_{k j} z_j+z_k & =b_k, & z_k & =b_k-\sum_{j=1}^{k-1} l_{k j} z_j
\end{array}
$$

The triangular system $U \mathbf{x}=\mathbf{y}$ yields similar equations, but in reverse order:
$$
\left[\begin{array}{ccccc}
u_{11} & u_{12} & u_{13} & \cdots & u_{1 n} \\
0 & u_{22} & u_{23} & \cdots & u_{2 n} \\
0 & 0 & u_{33} & \cdots & u_{3 n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & u_{n n}
\end{array}\right]\left[\begin{array}{c}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n
\end{array}\right]=\left[\begin{array}{c}
z_1 \\
z_2 \\
z_3 \\
\vdots \\
z_n
\end{array}\right] 
$$ 

$$
\begin{aligned}
u_{n,n}x_n = z_n \quad  & x_n = \frac{1}{u_{n, n}} z_n   \\ 
u_{n-1, n-1} x_{n-1}+u_{n-1, n} x_n=z_{n-1}, \quad & x_{n-1}=\frac{1}{u_{n-1, n-1}}\left(z_{n-1}-u_{n-1, n} x_n\right),  \\
\vdots \quad & \vdots \\
\sum_{j=k}^n u_{k j} x_j=z_k, \quad &  x_k=\frac{1}{u_{k k}}\left(z_k-\sum_{j=k+1}^n u_{k j} x_j\right) 
\end{aligned}
$$

All the above calculations assumed that we know the LU factorization of $A$. However, we do not know the LU factorization of $A$ in general. We can compute the LU factorization of $A$ using the __Gaussian elimination__ method.


A matrix of the form $G_1$ is called a Gauss transformation. This is a simpler
linear transformation that zeros out entries in a matrix. A matrix of this 
form is also called an atomic triangular matrix, or Frobenius matrix. 

For instance, 

$$
GA = \begin{bmatrix}
  1 & 0 & 0 & 0\\
  -4/3 & 1 & 0 & 0\\
  0 & 0 & 1 & 0\\
  0 & 0 & 0 & 1
\end{bmatrix} 
\begin{bmatrix}
  3\\
  4\\
  5\\
  6
\end{bmatrix} = \begin{bmatrix}
  3\\
  0\\
  5\\
  6
\end{bmatrix}
$$


If we repeat the above construction to have $G_1, G_2, \cdots, G_n$ for 
the __square__ matrix $A$ ($n \times n$), we can have matrix $U$. 

All the matrices $G_i$ are lower-triangular, and at the end of the process
this gives us an upper-triangular $U$:

$$G_{n-1} \cdots G_1 A = U$$

Formally, we can write 

$$A = G_1^{-1} \cdots G_{n-1}^{-1} U = LU$$

Since each $G_i$ is lower-triangular, the matrix $G_1^{-1} \cdots G_{n-1}^{-1}$
must be lower-triangular. This is our __LU factorization__. This kind of 
matrix decomposition will _transfer_ $A$ into _LU_ form after many iteration:

$$
A^{(\text{iteration})} = [L, U ]
$$

__Writing down the LU factorization__. We can simplify things quite a bit 
and find a nice expression for $L$. Consider two Gauss transformations
$G_i$ and $G_j$, which we can write out in the form

$$G_i = I + g_i e_i^T, \quad G_j = I + g_j e_j^T$$

where $e_i$ is a vector of zeros, with a $1$ at index $i$, and 

$$
g_i = \begin{bmatrix}
\vdots \\
(g_i)_{i+1} \\
\vdots \\
(g_i)_n \end{bmatrix}
$$

Suppose that $j \leq i$ and that $G_j$ and $G_i$ are Gauss transformations 
as above. Then 

$$G_j G_i = (I + g_j e_j^T) (I + g_i e_i^T) = I + g_j e_j^T + g_i e_i^T$$

For example, taking the inverse of $G_i$ is easy:

$$(G_i)^{-1} = I - g_i e_i^T$$


__A first implementation__. To implement this algorithm, following the
diagram above, we can loop over the columns of $A$. When operating on 
the $k$-th column of $A$, we can compte the entries of $L$ as:

$$
l_{ik} = \frac{a_{ik}^{(k)}}{a_{kk}^{k}}
$$

where $A^{(k)} = G_{k-1}\cdots G_1 A$ is the current iterate. Next we need 
to obtain $A^{(k+1)} = G_k A^{(k)}$. We only need to update the __lower-right corner__. 
The formula for row $i > k$ is given by 

$$
A^{(k+1)}[i, :] = A^{(k)}[i, :] - l_{ik} A^{(k)}[k, :]
$$

Now, let's implement this algorithm.


```python
def lu_outer_product(A: jnp.ndarray):
    """
    Implement outer product form of the LU factorization
    A: n by n matrix (square matrix)
    """
    n = A.shape[0]
    # loop over columns 
    for k in range(n):
        # the diagonal term can not be zero
        # otherwise the matrix cannot be factorized 
        assert A[k, k] != 0 
        # compute the entries of L 
        for i in range(k+1, n):
            # loop over row entries ([i, k]-row i column k)
            A = A.at[i, k].set(A[i, k]/A[k, k])
        
        # Outer-product of column k and row k 
        for i in range(k+1, n):
            for j in range(k+1, n):
                # subtract the outer-product
                temp = A[i, j] - A[i, k] * A[k, j]
                A = A.at[i, j].set(temp)
                
    return A
```

In the end, we will have the updated $A$ matrix, which is the upper-triangular having the following format

$$
A = [L, U]
$$

A __zero pivot__ occurs if $A[k, k]$ is zero at any step $k$ in the 
algorithm. In this case, the LU algorithm would try to divide by zero
and break. 

How can we fix this? To get intuition, consider the following example:

$$
\begin{bmatrix}
  1 & 0 & 0 & 0\\
  0 & 1 & 0 & 0\\
  1 & 0 & 0 & 1\\
  0 & 0 & 1 & 0\\
\end{bmatrix}
$$

We could __permute__ rows 3 and 4 in $L$, then $L$ becomes lower-triangular 
with $1$ on the diagonal. This suggests the remedy, which is to find 
an appropriate row permutation such that this problem can be avoided. However,
this will not solve all problems.

__Very small pivots, and the instability of the LU algorithm__. Consider
the  following matrix, for some value value $\varepsilon > 0$

$$
\begin{bmatrix}
  \varepsilon & 1\\
  1 & \pi \\
\end{bmatrix}
$$

After LU factorization, we should have:

$$
L = \begin{bmatrix}
   1 & 0 \\
  \varepsilon^{-1} & 1 \\
\end{bmatrix}, \quad U = \begin{bmatrix}
  \varepsilon & 1\\
  1 & \pi-\varepsilon^{-1} \\
\end{bmatrix}
$$

```python
# set \varepsilon = 10^[-14]
varep = 10e-14
test_b = jnp.array(
    [
        [varep, 1],
        [1, jnp.pi]
    ]
)
test_b

# Array([[9.9999998e-14, 1.0000000e+00],
#        [1.0000000e+00, 3.1415927e+00]], dtype=float32)

lu = lu_outer_product(test_b)
lu

# Array([[1.0000000e+00, 0.0000000e+00],
#        [1.0000001e+13, 1.0000000e+00]], dtype=float32)

L = jnp.array(
    [
        [1.0, 0.0],
        [1.0000001e+13, 1.0]
    ]
)

U = jnp.array(
    [
        [9.9999998e-14, 1.0],
        [0.0, -1.0000001e+13]
    ]
)

jnp.allclose(L@U, test_b)

# False
```


From the above calculation, we can see 

$$
\begin{aligned}
A & = \begin{bmatrix}
  9.9999998e-14 & 1.0000000e+00\\
  1.0000000e+00 & 3.1415927e+00\\
\end{bmatrix}  \neq L U  \\ 
LU & = \begin{bmatrix}
  1.0000000e+00 & 0.0000000e+00\\
  1.0000001e+13 & 1.0000000e+00\\
\end{bmatrix} \begin{bmatrix}
  9.9999998e-14 & 1.0000000e+00\\
  0.0000000e+00 & -1.0000001e+13\\
\end{bmatrix} \\ 
& = \begin{bmatrix}
  9.9975583e-14 & 1.0000000e+00\\
  9.9962425e-01 & 0.0000000e+00\\
\end{bmatrix}
\end{aligned}
$$


__Condition number of a matrix.__ The condition number of a matrix $A$ is 
fundamental to understanding the accuracy of numerical calculations involving
$A$.


I should point out that there are many different condition numbers. In general, a condition
number applies not only to a particular matrix, but also to the problem being solved. Are
we solving linear equations, inverting a matrix, finding its eigenvalues, or computing
the exponential? A matrix can be poorly conditioned for inversion while the eigenvalue
problem is well conditioned. Or, vice versa.

However, when we simply say a matrix is “ill-conditioned”, we are usually just thinking
of the sensitivity of its inverse, i.e. of the _condition number for inversion_, and not of all the
other condition numbers.

__Norms__. In order to make the sensitivity notions more precise, let’s 
start with a vector norm. Specifically, with the _Euclidean norm or 2-norm_:

$$||x|| = \left ( \sum_i x_i^2 \right )^{1/2}$$


The corresponding norm of a matrix $A$ measures how much the mapping induced by that
matrix can __stretch__ vectors.

$$M = ||A|| \equiv \max \frac{||Ax||}{||x||}$$

It is sometimes also important to consider how much a matrix can shrink vectors.

$$m = \min \frac{||Ax||}{||x||}$$

The reciprocal of the minimum stretching is the norm of the inverse, because 
when $Ax = b$

$$m = \min \frac{||Ax||}{||x||} =  \min \frac{||b||}{||A^{-1}b||} = \frac{1}{\max \frac{||A^{-1}b||}{||b||}} = \frac{1}{||A^{-1}||}$$


A _singular_ matrix is one that can map nonzero vectors into the zero vector.
For a singular matrix 

$$m = 0$$

and the inverse does not exsit. 

The ratio of the maximum to minimum stretching is the condition number for 
inversion

$$\kappa(A) \equiv \frac{M}{m}$$


We explained and condition number from the perspective of linear system
$Ax = b$ by exploring how much the matrix $A$ can stretch the vector. We have
also said that the condition number can be calculated by 

$$\kappa(A) = \frac{\sigma_1}{\sigma_n}$$

The following figure gives the connection between two expressions of 
condition number.


<div class='figure'>
    <img src="/math/images/linear-system-decomposition.png"
         alt="floating number illustrated"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The illustration of the linear system decomposition.
    </div>
</div>


The condition number $\kappa(A)$ is involved in the answer to the 
question: _how much can a change in the right hand side of a system of simultaneous linear equations affect the solution_?

Consider a system of equations:

$$Ax = b$$

and a second system obtained by altering the right-hand side:

$$A(x + \delta x) = b + \delta b$$


Think of $\delta b$ as being the error (forward error) in $b$ and $\delta x$
being the resulting error (backward error) in $x$, although we need to make 
any assumptions that the errors are small. Because $A(\delta x) = \delta b$,
the definition of $M$ and $m$ immediately lead to 

$$||b|| \leq M ||x||, \quad ||\delta b|| \geq m ||\delta x||$$

Consequently, if $m \neq 0$, 

$$ \frac{||\delta x||}{||x||} \leq  \kappa(A) \frac{||\delta b||}{||b||}$$

The quantity $||\delta b|| / ||b||$ is the relative change (relative forward error)
in the right-hand side, and the quantity $||\delta x|| / ||x||$ is the 
resulting relative change (relative backward error) in the solution. The 
advantage of using relative changes is that they are _dimensionless_ - they 
are not affected by overall scale factors. 


This inequality shows that the condition number is a relative error 
magnification factor. __changes in the right-hand side can cause changes $\kappa(A)$ times as large in the solution__. 

- If $\kappa(A) > 10^8$, with `float64`, the result of the calculation may gradually become inaccurate. 

- For $\kappa(A) > 10^{16}$, the solution can become completely inaccurate.

- There are cases when $\kappa(A)$ is very large but the numerical solution of $Ax=b$ can still be computed accurately.
For example if $A$ is diagonal, $x$ can be computed with machine accuracy regardless 
of the condition number of $A$.


## LU factorization with pivoting

Finally, armed with our new understanding of floating-point arithmetic
and error analysis, we can fix the issues with our previous of LU factoriztion. 

__Partial pivoting__. The LU algorithm with partial pivoting, is a 
modification of our LU decomposition algorithm where we perform
a row swap at each step $k$ so that the __absolute largest__ entry 
in $A[k:n, k]$ appears in the $(k, k)$ position. 

```python
def lu_partial_pivoting(A: jnp.ndarray):
    """
    LU factorization with partial pivoting
    A is a n by n matrix (square)
    """
    n = A.shape[0]
    # make sure the data type should be float 
    A = A.astype(jnp.float32)
    # permutation matrix
    P = jnp.eye(n, dtype=jnp.float32)
    for k in range(n):
        # find the pivot index
        pivot_idx = jnp.argmax(jnp.abs(A[k:, k])) + k
        # swap row k with the row having the absolute largest entry
        for j in range(n):
            # update columns element-wise 
            temp = A[pivot_idx, j]
            A = A.at[pivot_idx, j].set(A[k, j])
            A = A.at[k, j].set(temp)

        # update permutation matrix P
        # index with [[]], access to the whole row
        P = P.at[[k, pivot_idx], :].set(P[[pivot_idx, k], :])

        # compute the entries of L 
        for i in range(k+1, n):
            # loop over row entries ([i, k]-row i column k)
            A = A.at[i, k].set(A[i, k]/A[k, k])

        # doing LU factorization
        for i in range(k+1, n):
            for j in range(k+1, n):
                # subtract the outer-product
                temp = A[i, j] - A[i, k] * A[k, j]
                A = A.at[i, j].set(temp)
    
    return P, A
```

The permutation matrix $P$ might be different when you have 
different kinds of algorithm to do LU factorization. The above
algorithm obtains a decomposition of the form

$$PA = LU$$

where $P$ is a permutation matrix, $L$ is lower-triangular, and $U$
is upper-triangular. 

It is saying that there is a permutation $P$ such that if we perform
the $LU$ factorization on $PA$, with no pivoting, then the largest
entry in the column $k$ is always on the diagonal. 


LU factorization works for the square matrix and
the matrix should have the full rank. How about those matrices that do not have
the full rank?

__Rank-revealing factorization__. We say that a factorization of a rank-$r$
matrix $A$ as $A = WV^T$ where $W$ and $V$ have $r$ columns is __rank revealing__.
Such a factorization reveals that the rank of $A$ is $r$. 

Suppose that $A$ has rank $r$, and that the first $r$ columns and $r$ rows
of $A$ are linearly independent. It turns out that if we run our $LU$ 
algorithm, at step $r$, we will obtain a low-rank factorization! It 
looks like this: 

<div class='figure'>
    <img src="/math/images/lu-reveal-rank.png"
         alt="floating number illustrated"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The illustration of the LU factorization with rank revealing.
    </div>
</div>

Our LU factorization will only work if $\text{det}(A[1:r, 1:r]) \neq 0$, 
which may not be the case in general. 

How do we proceed? The solution is to add _column pivoting_ as well as 
_row pivoting_: we need to make sure that linearly dependent columns 
end up at the end. 

__Full pivoting.__ The LU algorithm with full pivoting, illustrated 
in the above figure, is a modification of our LU decomposition
algorithm where we perform a row swap and a column swap at each
step $k$ so that the largest entry in the $A[k:n, k:n]$ block 
appears in the $(k, k)$ position. 

The factorization produced by the full pivoting algorithm is of 
the form 

$$PAQ^T = LU$$

If at any point in the process, the remaining block in $A$ is 
filled only with zeros, we can stop the algorithm, and we 
have arrived a rank-revealing factorization. 


__Rook pivoting.__ The full pivoting algorithm works well, but it can get
very expensive because of the need to search for the largest entry. Actually,
finding the largest entry is not required. We just need a "large enough" 
pivot. For this purpose, _rook pivoting_ is a good strategy. 


<div class='figure'>
    <img src="/math/images/rook-pivoting.png"
         alt="floating number illustrated"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The illustration of the LU factorization with rook pivoting strategy.
    </div>
</div>

The LU algorithm with rook pivoting, illustrated in the above figure, 
is a modification of our LU decomposition algorithm, where at step $k$
we find any entry of $A$ that's maximal in both its row and its column
and perform a row and column swap to put that entry in the $(k, k)$ position. 


```python
def lu_rook_pivoting(A: jnp.ndarray):
    """
    LU factorization with rook pivoting
    """
    n = A.shape[0]
    # make sure the data type should be float 
    A = A.astype(jnp.float32)
    # row permutation matrix
    P = jnp.eye(n, dtype=jnp.float32)
    # column permutation matrix
    Q = jnp.eye(n, dtype=jnp.float32)

    for k in range(n):
        # rook pivoting
        # initialiye row and col index for while loop 
        row0, row_idx, col0, col_idx = 1, 0 , 1, 0
        while row_idx != row0 or col_idx != col0:
            # save the old values 
            row0, col0 = row_idx, col_idx 
            # get the row k
            row_pivot = jnp.abs(A[k+row_idx, k:])
            # search the pivot (largest) in the row k
            col_idx = jnp.argmax(row_pivot)
            # get the column with the largest value
            col_pivot = jnp.abs(A[k:, col_idx+k])
            # search the pivot in the col 
            row_idx = jnp.argmax(col_pivot)
            # the while loop will stop when search is finished 

        # now we have pivot index row_idx and column_idx
        
        # first, we update row_idx and col_idx with step k
        row_idx += k
        col_idx += k 

        # update row and column permutation matrix
        P = P.at[[k, row_idx], :].set(P[[row_idx, k], :])
        Q = Q.at[:, [k, col_idx]].set(Q[:, [col_idx, k]])

        # swap rows 
        for j in range(n):
            temp = A[row_idx, j]
            A = A.at[row_idx, j].set(A[k, j])
            A = A.at[k, j].set(temp)
        # swap columns
        for i in range(n):
            temp = A[i, col_idx]
            A = A.at[i, col_idx].set(A[i, k])
            A = A.at[i, k].set(temp)

        # perform LU factorization

        if A[k, k] != 0:
             # compute the entries of L 
            for i in range(k+1, n):
                # loop over row entries ([i, k]-row i column k)
                A = A.at[i, k].set(A[i, k]/A[k, k])

            # doing LU factorization
            for i in range(k+1, n):
                for j in range(k+1, n):
                    # subtract the outer-product
                    temp = A[i, j] - A[i, k] * A[k, j]
                    A = A.at[i, j].set(temp)
                
    return P, Q, A
```

The _rank-revealing factorization_ gives the different $LU$ factorization
as we also performed the column permutation. In the end, we have 

$$A = WV^T$$

This algorithm will be helpful for us to understand the QR decomposition later. 


## Cholesky factorization 

The LU factorization can be made more efficient for many classes of 
special matrices that have special properties. An important such 
class are _symmetric positive definite matrices_ (SPD). 

A symmetic matrix is _symmetric positive definite_ (SPD) if $x^t A x > 0$
for all non-zero vectors $x$, or equivalently, if all of the eigenvalues
are strictly positive. 

It is natural to think if a matrix is symmetric,
we should be able to speed things up. 


__No zero pivots for SPD matrices.__ If $A$ is SPD, the non-pivoting
LU factorization will never encounter a zero pivot. 

If $A$ is SPD and $B$ is non-singular, then $B^TAB$ is also SPD:

$$x^T(B^T AB)x = (Bx)^T A (Bx) > 0$$

Suppose $A$ looks like 

$$A = \begin{bmatrix} a & C^T \\ C & b \end{bmatrix}$$

If we perform one step of the LU factorization, we obtain

$$
A = \begin{bmatrix} a & C^T \\ C & b \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ C/a & I \end{bmatrix} \begin{bmatrix} a & C^T \\ 0 & B - (1/a)CC^T \end{bmatrix}
$$


Now, if we do futher decompositon, we could have 

$$
A = \begin{bmatrix} a & C^T \\ C & b \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ C/a & I \end{bmatrix} \begin{bmatrix} a & 0 \\ 0 & B - (1/a)CC^T \end{bmatrix} 
\begin{bmatrix} 1 & C^T/a \\ 0 & I \end{bmatrix}
$$

Since the matrix on the left and right are not singular, therefore

$$\begin{bmatrix} a & 0 \\ 0 & B - (1/a)CC^T \end{bmatrix} $$ 

must be SPD as well. This means $ B - (1/a)CC^T $ is SPD!


__Schur complement.__ The matrix 

$$ B - (1/a)CC^T $$ 

is called the _Schur complement_ of entry $a$ in $A$. More generally, 
assume $A$ is decomposed in a $2 \times 2$ block form:

$$A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} $$

Where all diagonal components are _square_ matrices. If we run Cholesky
for $k$ steps, we have the following decomposition:

$$A = \begin{bmatrix} I & 0 \\ A_{21} A_{11}^{-1} & I  \end{bmatrix} 
\begin{bmatrix} A_{11} & 0 \\ 0 & A_{22} - A_{21} A_{11}^{-1} A_{12} \end{bmatrix} 
\begin{bmatrix} I & A_{11}^{-1} A_{12} \\ 0 & I \end{bmatrix} $$


The square block of size $n-k$ in the bottom right, 

$$A_{22} - A_{21} A_{11}^{-1} A_{12} $$

is called the Schur complement of $A_{11}$ for matrix $A$. If $A$ is SPD,
then the Schur complement of $A_{11}$ is also SPD. 


__Symmetric factorization.__ When the matrix $A$ is SPD, it has a special 
form illustrated in the following figure. 


<div class='figure'>
    <img src="/math/images/spd-illustration.png"
         alt="floating number illustrated"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The illustration of the LU factorization for a SPD matrix.
    </div>
</div>


That is, we get a decomposition of the form 

$$A = LDL^T$$ 

where $D$ is a diagonal matrix with positive entries on the diagonal. 
This is also known as the _Cholesky factorization_. 


__Cholesky factorization.__ It is common to rewrite this factorization
$A = LDL^T$ in a different form. Since the diagonal entries of $D$
are positive, we can define

$$G = LD^{1/2}$$

from which we obtain a symmetric factorization: 

$$A = G G^T$$


```python
def cholesky_lu(A: jnp.ndarray):
    """
    A is symmetric postive definite matrix
    """
    n = A.shape[0]
    A = A.astype(jnp.float32)
    for k in range(n):
        # doing outer product
        for i in range(k, n):
            for j in range(k):
                # subtract the outer-product
                temp = A[i, k] - A[i, j] * A[k, j]
                A = A.at[i, k].set(temp)

        a_kk = jnp.sqrt(A[k, k])

        for i in range(k, n):
            temp = A[i, k]/a_kk
            A = A.at[i, k].set(temp)

    return A


def cholesky_lu2(A: jnp.ndarray):
    """
    Cholesky lu factorization with row operation (vectorized version)
    """ 
    n = A.shape[0]
    A = A.astype(jnp.float32)
    for k in range(n):
        for i in range(k+1, n):
            # update i-th row of upper Cholesky 
            row_k = A[k, i:] * A[k, i] / A[k, k]
            A = A.at[i, i:].set(A[i, i:] - row_k)
        
        # normalize k-th row of upper Choleksy factor
        A = A.at[k, k:].set(A[k, k:] / jnp.sqrt(A[k, k]))

    return A
```






All figures used in this post are from the book by {% cite darve2021numerical %}. 













{% endkatexmm %}