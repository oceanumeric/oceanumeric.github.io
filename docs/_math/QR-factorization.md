---
title: QR Factorization
subtitle: A QR factorization is a factorization of a matrix A into a product A = QR of an orthogonal matrix Q and an upper triangular matrix R. This kind of decomposition is useful in solving linear least squares problems and in the eigendecomposition of a matrix, which shows the structure of the matrix in terms of its eigenvalues and eigenvectors.
layout: math_page_template
date: 2023-04-02
keywords: QR factorization, orthogonal matrix, upper triangular matrix, linear least squares, eigendecomposition, eigenvalues, eigenvectors householder reflections, givens rotations
published: true
tags: algorithm data-science machine-learning numerical-linear-algebra python
---

QR factorization is a factorization of a matrix A into a product A = QR of an orthogonal matrix Q and an upper triangular matrix R. Since it reveals very useful information about the matrix A, it is a very important factorization. It is used in solving linear least squares problems and in the eigendecomposition of a matrix, which shows the structure of the matrix in terms of its eigenvalues and eigenvectors.

- [Definition](#definition)
- [Householder reflections](#householder-reflections)
- [Givens rotations](#givens-rotations)
- [Upper Hessenberg Form](#upper-hessenberg-form)


## Definition


{% katexmm %}

Recall that a matrix $Q$ is an _orthogonal matrix_ if $Q^T Q = I$. If $Q$ 
is orthogonal, then $||Qx||_2 = ||x||_2$ for all $x$; that is, $Q$ 
does not change the length of vectors. The operation that don't change 
the length of vectors are rotations and reflections, so an orthogonal 
matrix can then be thought of as a map that combines rotations and reflections. 

__QR factorization.__ Let $A$ be an $m \times n$ __real__ matrix where
$m \geq n$. Then there is an orthogonal matrix $Q$ and an upper-triangular
matrix $R$ so that $A = QR$. This is called the __QR factorization__. 

When $A$ is a __complex__ matrix, we can still write $A = QR$, where 
$Q$ is unitary instead of orthogonal. 

For the rest of this chapter, we will assume that $A$ is real, but it's 
important to know that $QR$ decomposition works for complex matrices 
as well. 

The $QR$ decomposition is a fundamentally important matrix factorization. It
is straightforward to implement, is numerically stable, and provides the 
basis of several important algorithms. In this lab we explore several 
ways to produce the QR decomposition and implement a few immediate 
applications.

The QR decomposition of a matrix $A$ is a factoration 

$$A =QR$$ 

where $Q$ has orthonormal columns and $R$ is upper triangular. Every $m \times n$
matrix $A$ of rank $n \leq m$ has a QR decomposition, with two main forms. 

- Reduced QR: Q is $m \times n$, R is $n \times n$, and the columns $\{q_j\}_{j=1}^n$ 
of Q form an orthonormal basis of $A$ 

-Full QR: Q is $m \times m$, R is $m \times n$. In this case, the columns
$\{q_j\}_{j=1}^m$ of Q form an orthonormal basis for all of $F^m$, and the last $m-n$ rows of R only contain zeros. 

We distinguish between these two forms by writing $\hat{Q}$ and $\hat{R}$
for reduced decomposition and Q and R for the full decomposition. 


<div class='figure'>
    <img src="/math/images/QR-factorization1.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of QR factorization.
    </div>
</div>

Different forms of the QR factorization, depending on the matrix dimensions, 
are shown in the following figure:



<div class='figure'>
    <img src="/math/images/QR-factorization2.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The illustration of QR factorization.
    </div>
</div>


The QR factorization has many applications: 

1. The QR factorization can be used to solve least-squares problems,
that is, problems of the form $\min_x ||Ax - b||_2$, where $A$ is 
a tall skinny matrix. 

2. The QR factorization is used as part of the eigenvalue and 
singular value algorithms. We'll see these in Chapter 5. 

3. The QR factorization is also used to in iterative methods to 
solve linear system and compute eigenvalues (e.g., Krylov methods). We'll
see these in Chapter 6. 

__Least-squares problems.__ To see why the QR decomposition might be useful
(and to get a taste of what's to come in section 4.4 below), let's look 
briefly at the _least-squares problem_. let $A \in \mathbb{R}^{m \times n}, m \geq n$.
We'd like to find the $x$ such that $Ax$ is closest to $b$ in Euclidean distance.
That is,

$$x^* = \arg \min_x ||Ax - b||_2$$


To do this, we'll use the QR factorization (with $Q$ square):

$$||Ax - b||_2  = ||Q^T (Ax - b)||_2 = ||Q^T(QRx - b) ||_2 = ||Rx - Q^Tb||_2$$

Above, we used the fact that for any orthogonal matrix $Q$ and any vector
$x$, we have 

$$||Qx||_2 = ||x||_2$$

This is because

$$||Qx||_2^2 = (Qx)^T(Qx) = x^T Q^T Q x = x^T x = ||x||_2^2$$


Finding $x$ that minimizes $||Rx - Q^Tb||_2$ turns out to be much easier 
than finding $x$ that minimizes $||QRx - b||_2$. 


## Householder reflections

First, let's consider the QR decomposition when $Q$ is square. One 
of the most reliable methods for calculating $QR$ factorization in 
this case is to use _Householder reflections_. This method is computationally
efficient, robust, and accurate. In many ways, the idea behind this 
approach is similar to the LU factorization. For $A = LU$, we rewrote 
the equation as 

$$L^{-1}A = U$$

That is, we asked the question: is there a lower-triangular matrix that 
can transform $A$ into an upper-triangular matrix? 

Now we ask: is there an orthogonal orthogonal matrix that can transform 
$A$ into a upper-triangular matrix:

$$Q^T A = R $$

As before, our goal is to create zeros below the diagonal, and we begin with
the first column. 


<div class='figure'>
    <img src="/math/images/householder1.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The illustration of Householder reflections step 1.
    </div>
</div>


__Householder reflection.__ We need to apply an orthogonal transformation
$Q_1^T$ to transform the first column into a vector in the direction 
of $e_1$ (meaning $x$-axis in the dimension of two). Let's write 

$$A = [a_1 | \cdots |a_n]$$

Then $Q_1^T a_1$ should be parallel to $e_1$. Since $Q_1$ does not change 
the norm of $a_1$, we must have


<div class='figure'>
    <img src="/math/images/householder2.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The illustration of Householder reflections step 2.
    </div>
</div>


How should we choose $Q_1$? One logical choice would be a rotation that maps
$a_1$ parallel to $e_1$; however, it turns out that rotations in high dimensions
are not so easy to set up. Thus, we will instead to choose $Q_1$ to be a 
_reflection_ that maps $a_1$ parallel to $e_1$. The picture looks like this:

<div class='figure'>
    <img src="/math/images/householder3.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The illustration of Householder reflections.
    </div>
</div>


Now that we have an idea of what the reflection should be doing, let's figure
out a mathematical formula to capture it. 


To begin, we suppose that we can find the _hyperplane_ and its orthogonal 
vector $v$. And this reflection could be defined by this vector $v$. If we have
the vector $v$ and its corresponding hyperplane that sits in the middle of 
$x$ and $e_1$, we could 

- project $x$ onto $v$ and get

$$y = \frac{v^Tx}{v^Tv} v$$

- Then traverse to the vector $e_1$ by add $-2y$ to $x$

$$x-2y = (I - \frac{vv^T}{v^Tv}) x $$


<div class='figure'>
    <img src="/math/images/householder4.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The illustration of Householder reflections.
    </div>
</div>

__Reflections.__ Let $H$ be the matrix which represents reflection over 
the hyperplane orthogonal to some vector $v$. Then $H$ is given by

$$H = I - \beta v v^T, \quad \text{with} \quad \beta = \frac{2}{v^Tv}$$

We need to __pick v correctly__ to arrive at hour Householder reflections.
We need to find a formula for $v$ that will reflect a given vector $x$ 
onto $||x||e_1$. The following geometric argument shows that 

$$v = x- ||x||e_1$$

will do the trick. 


<div class='figure'>
    <img src="/math/images/householder5.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 6.</span> The illustration of Householder reflections.
    </div>
</div>


To see this formally, recall that we have 

$$Hx  = (I - \beta v v^T) x = x - \beta (v^T x) v$$

Therefore, we conclude 

$$\beta(v^T x) v = x - ||x||e_1$$

Since the norm of $v$ does not matter and $v$ should parallel to $x - ||x||e_1$,
then we can just pick

$$v = x- ||x|| e_1$$


__Householder reflections.__ The __Householder reflection__ that maps 
$x$ to $||x||e_1$ is given by 

$$H = I - \beta v v^T$$

where $v = x - ||x||e_1$ and $\beta = \frac{2}{v^Tv}$.

Lemma: For $x \in \mathbb{R}^n$, if $v=x+\|x\| e_1$ is nonzero, then $H_v(x)=-\|x\| e_1$, and if $v=x-\|x\| e_1$ is nonzero, then $H_v(x)=\|x\| e_1$.

Proof. The argument for the two cases is similar; we show the argument for $v=x+$ $\|x\| e_1 \neq 0$. For $x_1$ the first entry of $x$ we have
$$
\begin{aligned}
v^T v & =\left(x+\|x\| e_1\right)^T\left(x+\|x\| e_1\right) \\
& =x^T x+\|x\| x^T e_1+\|x\| e_1^T x+\|x\|^2 e_1^T e_1 \\
& =2\|x\|^2+2\|x\| x_1=2\|x\|\left(\|x\|+x_1\right),
\end{aligned}
$$
and
$$
\begin{aligned}
v v^T x & =\left(x+\|x\| e_1\right)\left(x+\|x\| e_1\right)^T x \\
& =x x^T x+\|x\| x e_1^T x+\|x\| e_1 x^T x+\|x\|^2 e_1 e_1^T x \\
& =\|x\|^2 x+\|x\| x_1 x+\|x\|^3 e_1+\|x\|^2 x_1 e_1 \\
& =\|x\|\left(\|x\| x+x_1 x+\|x\|^2 e_1+\|x\| x_1 e_1\right) \\
& \left.=\|x\|\left(\|x\|+x_1\right) x+\|x\|\left(\|x\|+x_1\right) e_1\right) \\
& =\|x\|\left(\|x\|+x_1\right)\left(x+\|x\| e_1\right) .
\end{aligned}
$$
Then
$$
\begin{aligned}
H_v(x) & =x-2 \frac{v v^T x}{v^T v} \\
& =x-2 \frac{\|x\|\left(\|x\|+x_1\right)\left(x+\|x\| e_1\right)}{2\|x\|\left(\|x\|+x_1\right)} \\
& =x-\left(x+\|x\| e_1\right) \\
& =-\|x\| e_1 .
\end{aligned}
$$


__Roundoff errors.__ This is a practical issue. we have shown that the 
first component of $v$ is 

$$v_1 = x_1 - ||x||$$

while component $i \geq 2$ is $x_i$. However, if $x_1 > 0$ and $x_2^2 + \cdots + x_n^2 << x_1^2$, the subtraction $x_1 - || x||$ could lead to large cancellation errors.


large cancellation error or __catastropic cancellation__ is the phenomenon 
that subtracting good approximations to two nearby numbers many yield
a very bad approximation to the difference of the original numbers. 

_Remark_: we care relative error when it comes to cancellation error (the
absolute error might be small when you have 60 digits but the relative error
could be huge and it might change the results dramatically once those errors
were propagated, see [this](https://en.wikipedia.org/wiki/Catastrophic_cancellation))


Although this sounds like a remote possibility, in practice this happends 
very often when we use the $QR$ decomposition in the algorithm to compute
eigenvalues (which we cover in the next chapter). To fix this, in that case 
we can project $x$ onto $-e_1$; $v_1$ is then given by 

$$v_1 = x_1 + ||x||$$


This choice for $v_1$ does not lead to large roundoff errors when 
$x_1 > 0$. 


Before we implement the algorithm, let's summarize what we have done: 

- we have a vector $x$ and its projection on $e_1$ is $||x|| e_1$
- the hyperplane could be calculated by $x + ||x|| e_1$
- the vector $v$ is $x - ||x|| e_1$ because 

$$ (x - ||x|| e_1)^T (x + ||x|| e_1) = 0 $$ 

- because of the iusse of roundoff errors, we will project $x$ onto $-e_1$,
this means $v_1 = x_1 + ||x||$.

- _note_: we project $x$ onto different $e_1$ based on the sign of $x_1$


There is a special case we have to take it carefully. For a vector $x$, if

$$(x[1:n])^T (x[1:n]) = 0$$

since we do the following projection: 

$$
v = x - ||x|| e_1 = \begin{cases}
x + ||x|| e_1 & \text{if} \ x_1 > 0 \\
x - ||x|| e_1 & \text{if} \ x_1 < 0 \\
0 & \text{if} \ x_1 = 0 
\end{cases}
$$

It is not difficcult to show that $v^T v = 0 $ (meaning $\beta = 0$) whenever $(x[1:n])^T (x[1:n]) = 0$. 


```python
def house(x: jnp.ndarray):
    """
    Householder projection for the vector x 
    It computes beta and v for the Householder reflection:
    P = I - beta v v^T 

    Parameters
    ----------
    x: jnp.ndarray, shape (n, ) or (n, 1)

    Returns
    ----------
    beta: scalar 
    v: jnp.ndarray, shape (n, 1)
    Note: the algorithm from Golub and VanLoan is slight different
    as they normalize V into [1, v_2, ..., v_n]
    e.g. https://fa.bianp.net/blog/2013/householder-matrices/
    """

    x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)
    v = x.copy()

    # calculate the sigma
    sigma = jnp.linalg.norm(x[1:, 0]) ** 2
    # if sigma == 0, it means x is alread on e1
    if sigma == 0:
        beta = 0 
        # since x is alread on e1 we could just return its copy
        return beta, v 
    # calculate ||x||
    x_norm = jnp.sqrt(x[0, 0]**2 + sigma)

    if x[0, 0] > 0:
        # update v 
        v = v.at[0, 0].set(x[0, 0] + x_norm)
    else:
        v = v.at[0, 0].set(x[0, 0] - x_norm)

    beta = 2.0  / (v[0, 0]**2 + sigma)

    return beta, v


key = random.PRNGKey(333)
size = 5
x = random.randint(key, (size, ), -10, 10)
x
# Array([-3,  4, -4,  5, -9], dtype=int32)
beta, v = house(x)
print(beta, '\n', v)
# 0.0054533635 
#  [[-15.124355]
#  [  4.      ]
#  [ -4.      ]
#  [  5.      ]
#  [ -9.      ]]
```

Now we can verify 

$$\beta (v^T x) = 1, \quad Hx = ||x|| e_1$$


```python
beta * (v.T @ x)
# Array([0.9999999], dtype=float32)
H = np.eye(size) - beta * np.dot(v, v.T)
print(np.round(H.dot(x) / np.linalg.norm(x), decimals=15))
# [ 1.  0. -0.  0.  0.]
# test the accuracy of house
e1 = jnp.array([1.0, 0, 0, 0])
e1
# Array([1., 0., 0., 0.], dtype=float32)
beta, v = house(2*e1)
print(beta, '\n', v)
# 0 
#  [[2.]
#  [0.]
#  [0.]
#  [0.]]
H = np.eye(4) - beta * np.dot(v, v.T)
print(np.round(H.dot(e1) / np.linalg.norm(e1), decimals=15))
# [1. 0. 0. 0.]

def gvsn(n):
    """
    Generate a random vector with small norm
    """
    x = np.random.normal(size=n)
    x -= (x.mean()/2)
    return x / 1000

r = gvsn(4)
r
# array([ 0.0001777,  0.0003931, -0.0003471,  0.0017381])
r = jnp.array([0.0001777,  0.0003931, -0.0003471,  0.0017381])
np.linalg.norm(r)
# 0.0018241642
# e1 + r (a random vector with a very norm)
x = e1 + r
x 
# Array([ 1.0001777,  0.0003931, -0.0003471,  0.0017381], dtype=float32)
beta, v = house(x)
print(beta, '\n', v)

# 0.49982107 
#  [[ 2.0003572]
#  [ 0.0003931]
#  [-0.0003471]
#  [ 0.0017381]]
H = np.eye(4) - beta * np.dot(v, v.T)
print(np.round(H.dot(x) / np.linalg.norm(x), decimals=15))

# [-0.9999999  0.        -0.         0.       ]
```

For $x = e_1 +r$ where $r$ is a random vector with a very small norm. In
this case, we have $x_1 > 0$ and $x_2^2 + \cdots + x_n^2 << x_1^2$,
to avoid the large cancellation errors, we project $x$ onto $-e_1$, which is

$$
\begin{bmatrix}
-0.9999999 &   0 &       -0    &     0      
\end{bmatrix}
$$

In summary, we could 

```python
def house2(x: jnp.ndarray):
    """
    Householder projection for the vector x 
    It computes beta and v for the Householder reflection:
    P = I - beta v v^T 

    Parameters
    ----------
    x: jnp.ndarray, shape (n, ) or (n, 1)

    Returns
    ----------
    beta: scalar 
    v: jnp.ndarray, shape (n, 1)
    Note: the algorithm is based on Golub and VanLoan 5.1.1
    """

    x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)
    v = x.copy()

    # calculate the sigma
    sigma = jnp.linalg.norm(x[1:, 0]) ** 2
    # if sigma == 0, it means x is alread on e1
    if sigma == 0:
        beta = 0
    else:
        # calculate ||x||
        x_norm = jnp.sqrt(x[0, 0]**2 + sigma)

        if x[0, 0] <= 0:
            # update v 
            v = v.at[0, 0].set(x[0, 0] - x_norm)
        else:
            temp = - sigma / (x[0, 0] + x_norm)
            v = v.at[0, 0].set(temp)
    
        beta = 2.0 * (v[0, 0] ** 2) / (v[0, 0]**2 + sigma)
        temp = v / v[0, 0]
        v = v.at[:, :].set(temp)
        
    return beta, v


beta, v = house2(x)
print(beta, '\n', v)

# 1.6474086e-06 
#  [[    1.     ]
#  [ -238.57436]
#  [  210.65672]
#  [-1054.8616 ]]

H = np.eye(4) - beta * np.dot(v, v.T)
print(np.round((H @ x) / np.linalg.norm(x), decimals=15))

# [ 0.9999999 -0.         0.        -0.       ]
```


__Note__: The algorithm from Golub and VanLoan always project $x$ onto $e_1$ {% cite golub2013matrix %}.
In our post, we will simplify the following 

$$
v = x - ||x|| e_1 = \begin{cases}
x + ||x|| e_1 & \text{if} \ x_1 > 0 \\
x - ||x|| e_1 & \text{if} \ x_1 < 0 \\
0 & \text{if} \ x_1 = 0 
\end{cases}
$$

into 

$$ v = x + \text{sign}(x_1) ||x|| e_1$$

If we are dealing with a matrix $A$, then the first Householder vector should
be 

$$v^1 = a_1 + \text{sign}(a_{11}) ||a_1|| e_1$$


__Iterate this idea.__ In a manner similar to the LU factorization, we can
apply a series of Householder transformation to progressively reduce $A$ 
to upper-triangular form, with first zeroing entries in the first
column, the second column, etc. In the end, for $A \in \mathbb{R}^{m \times n}, m \geq n$, we have 

$$Q_{n-1}^T \cdots Q_1^T A = R$$

which is equivalent to 

$$A = Q_1 \cdots Q_{n-1} R = QR$$

This is our QR factorization. Note that $Q \in \mathbb{R}^{m \times n}$, and $R$
has zeros in the last $m-n$ rows if $m \geq n$. 

The code to implement this algorithm is given below. In the code, instead 
of allocating memory for $Q$ and $R$, we implicitly store the QR decomposition
like this:



<div class='figure'>
    <img src="/math/images/householder6.png"
         alt="floating number illustrated"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 7.</span> The illustration of Householder reflections.
    </div>
</div>

Here are the details of our algorithm, we need to do Householder transformation
on $A$ such as

$$HA = (I - \beta v v^T)A = A - \beta v (v^T A)$$


```python
def qr_factorization(A: jnp.ndarray):
    """
    QR factorization with Householder transformation

    Parameters
    ----------
    A: jnp.ndarray shape (m, n), assuming m >= n
        if m < n, we will transpose the matrix

    Returns
    ----------
    factor R
    A with factor R in upper-triangular part and
    Lower-triangular part of A is sequence of v vectors  
    """

    m, n = A.shape
    A = A.astype(jnp.float32)  # (m, n)

    if m < n:
        A = A.T
        m, n = A.shape

    # loop over columns as we will transform column vector
    for i in range(n):
        beta, v = house(A[i:, i])  # v.shape = (m-i, 1)
        # calculate beta v (v^T A)
        # A[i:, i:].shape = (m-i, n-i)
        # temp.shape = (m-i, n-i)
        temp = beta * v @ (v.T @ A[i:, i:])  
        # update A for R part
        A = A.at[i:, i:].set(A[i:, i:]-temp)
        # update A for Q part
        # saving v in the lower-triangular part of A
        # since v[0] = 1, there is no need to store it
        A = A.at[i+1:, i].set(v[1:, 0])
    
    return jnp.tril(A.T).T, A


A = jnp.array(
    [
        [1, -1, 4],
        [1, 4, -2],
        [1, 4, 2],
        [1, -1, 0]
    ]
)
A

# Array([[ 1, -1,  4],
#        [ 1,  4, -2],
#        [ 1,  4,  2],
#        [ 1, -1,  0]], dtype=int32)

np.linalg.qr(A)

# (array([[-0.5,  0.5, -0.5],
#         [-0.5, -0.5,  0.5],
#         [-0.5, -0.5, -0.5],
#         [-0.5,  0.5,  0.5]]),
#  array([[-2., -3., -2.],
#         [ 0., -5.,  2.],
#         [ 0.,  0., -4.]]))

qr_factorization(A)

# (Array([[-2.       , -3.       , -2.       ],
#         [ 0.       , -5.       ,  1.9999995],
#         [ 0.       ,  0.       , -4.       ],
#         [ 0.       ,  0.       ,  0.       ]], dtype=float32),
#  Array([[-2.       , -3.       , -2.       ],
#         [ 1.       , -5.       ,  1.9999995],
#         [ 1.       ,  3.3333333, -4.       ],
#         [ 1.       , -1.6666667, -3.2000003]], dtype=float32))

A = jnp.array(
    [
        [0.5, 0.903281 ,  1.10219 , 1.09724],
        [0.5, 0.520598 , -0.152935,  -0.767982],
        [0.5,  -0.0205981,  -0.513732,   0.267982],
        [0.5,  -0.403281,    0.231146 , -0.0972388]
    ]
)

R, Av = qr_factorization(A)
R 
# Array([[-1.       , -0.4999999, -0.3333346, -0.2500006],
#        [ 0.       , -0.9999994, -0.666668 , -0.5000008],
#        [ 0.       ,  0.       ,  1.0000012,  0.7500015],
#        [ 0.       ,  0.       ,  0.       , -0.9999991]], dtype=float32)

Q, R = jnp.linalg.qr(A)
R

# Array([[-0.9999999, -0.5      , -0.3333346, -0.2500006],
#        [ 0.       , -0.9999993, -0.6666681, -0.5000008],
#        [ 0.       ,  0.       ,  1.0000014,  0.7500015],
#        [ 0.       ,  0.       ,  0.       , -0.9999993]], dtype=float32)
```

You may notice that we do not explicitly calculate the Q part in our algorithm.
To calculate $Q$ we need a slightly different algorithm. Let's review 
our formula:

$$HA = (I - \beta v v^T)A = A - \beta v (v^T A)$$

where the vector $v$ is constructed by 

$$v = x \pm ||x|| e_1$$

where the sign was important for the first element as it determines
which direction ($e_1$ or $-e_1$) it will project $x$ onto.

__The scale of $v$ does not matter__. Therefore, we could normalize $v$
as unit vector then we have 

$$\beta = \frac{2}{v^Tv} = 2$$


```python
def qr_factorization2(A: jnp.ndarray) -> jnp.ndarray:
    """
    QR factorization with Householder transformation

    Parameters
    ----------
    A: jnp.ndarray, shape (m, n), assuming m >= n

    Returns
    ---------
    Q: Q factor, shape (m, m)
    R: R factor, shape (m, n)
    """
    m, n = A.shape
    A = A.astype(jnp.float32)
    R = A.copy()
    Q = jnp.eye(m)

    # construct sign function
    sign = lambda x: 1 if x >= 0 else -1

    # loop over columns
    for i in range(n):
        # construct column vector v from R based on formula
        # v = x + sign(x_1) ||x|| e_1
        # (as R is copy of A now) 
        v = R[i:, i].reshape(-1, 1)
        v_norm = jnp.linalg.norm(v)
        v = v.at[0, 0].set(v[0, 0] + sign(v[0, 0]) * v_norm)
        # normalize v
        v = v.at[:, :].set(v/jnp.linalg.norm(v))

        # update R and Q
        R = R.at[i:, i:].set(R[i:, i:] - 2 * v @ v.T @ R[i:, i:])
        Q = Q.at[i:, :].set(Q[i:, :] - 2 * v @ v.T @ Q[i:, :])

    
    return Q.T, R
```


## Givens rotations 

Householder transforms are the big hammer of orthogonal transforms. They
are efficient at creating a lot of zeros in a given column. However, 
consider the problem of computing the QR factorization of a matrix $A$
that looks like this:


<div class='figure'>
    <img src="/math/images/householder7.png"
         alt="floating number illustrated"
         style="width: 50%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 8.</span> The illustration of Householder reflections.
    </div>
</div>


That is, $A_{i, j}$ is zero for all indices such that $i > j +1$. (Such a matrix
is called an _upper Hessenberg_ matrix). In each step, we only need to zero
out a single entry (the one right below the diagonal), and the Householder
transforms are overkill. 

Since __we only need to zero out a single entry__, we need a small to do do
less transformation. It turns out that 2D rotation will work for upper 
Hessenberg matrix.

__Given rotations.__ Instead we need a small srewdriver that is adapted
to small jobs.

The problem can be reduced to considering a 2D vector $u = (u_1, u_2)$
and finding a rotation $G^T$ such that the vector becomes 
aligned with $+e_1$.

A __Given rotation__ which rotates $u = (u_1, u_2)^T$ to $||u|| e_1$ is the
$2 \times 2$ matrix defined by

$$ G^T = \begin{bmatrix} c & -s \\ s & c \end{bmatrix}, c = \frac{u_1}{||u||_2}, s = \frac{-u_2}{||u||_2}$$


```python
def givens(u1, u2):
    """
    Givens rotation for upper Hessenberg matrix

    Parameters
    ----------
    u1, u2 are float numbers (jnp.float32)
    
    Returns
    -------
    two float numbers c s 
    """

    if u2 == 0:
        c = 1 
        s = 0
    else:
        if jnp.abs(u2) > jnp.abs(u1):
            # this condition avoids potential overflows
            # when tau is large
            # we will explain this in the coming part
            tau = -u1/u2
            s = 1.0 / jnp.sqrt(1.0 + tau * tau)
            c = s * tau
        else:
            tau = -u2/u1
            c = 1.0/jnp.sqrt(1.0 + tau * tau)
            s = c* tau

    return c, s


def givens_transform(A:jnp.ndarray) -> jnp.ndarray:
    for i in range(A.shape[0]-1):
        c, s = givens(A[i, i], A[i+1, i])
        # construct Givens rotation matrix
        G = jnp.array(
            [
                [c, -s],
                [s, c]
            ]
        )
        # apply the Givens rotation to row k and k+1
        u = A[i:i+2, :].reshape(2, -1)
        A = A.at[i:i+2, :].set(G @ u)

    return jnp.round(A, 6)

key = random.PRNGKey(567)
A = random.uniform(key, (6, 6))
A = jnp.triu(A, -1)
A
# Array([[0.976706 , 0.8106253, 0.781171 , 0.6478064, 0.939659 , 0.7351937],
#        [0.2583095, 0.6409593, 0.1166685, 0.7197964, 0.7315263, 0.0051122],
#        [0.       , 0.0422235, 0.3764414, 0.3792816, 0.9099863, 0.4950393],
#        [0.       , 0.       , 0.2414137, 0.7459104, 0.2407993, 0.0194172],
#        [0.       , 0.       , 0.       , 0.2245039, 0.6656947, 0.5180018],
#        [0.       , 0.       , 0.       , 0.       , 0.9256719, 0.2474174]],      dtype=float32)
givens_transform(A)
# Array([[ 1.010163,  0.947512,  0.785125,  0.810408,  1.095229,  0.712242],
#        [ 0.0001  ,  0.414682, -0.048121,  0.56605 ,  0.557192, -0.131594],
#        [-0.000009,  0.000004,  0.452973,  0.671187,  0.854258,  0.442946],
#        [ 0.000005, -0.000002,  0.000114,  0.51102 ,  0.065007, -0.002208],
#        [ 0.000001, -0.000001,  0.000034,  0.000009, -1.166365, -0.548181],
#        [-0.000002,  0.000001, -0.000044, -0.000011, -0.000349,  0.30846 ]],      dtype=float32)
```

__Gram-Schmidt.__  Our starting point is that observation that, for every
$k$, the first $k$ columns $q_1, \cdots, q_k$ of $Q$ are an _orthonormal basis_
for the subspace spanned by $a_1, \cdots, a_k$.

Consider the Gram-Schmidt procedure, with  the vectors to be considered 
in the process as columns of the matrix $A$. That is 

$$
A = \begin{bmatrix}
\begin{array}{c|c|c|c}
a_1 & a_2 & \cdots & a_n
\end{array}
\end{bmatrix}
$$

Then, 

$$
\begin{aligned}
u_1 = a_1, & \quad e_1 = \frac{u_1}{||u_1||} \\ 
u_2 = a_2 - <a_2, e_1> e_1, & \quad e_2 =  \frac{u_2}{||u_2||} \\
\vdots 
\end{aligned}
$$ 

The  resulting QR factorization is 

$$
A = \begin{bmatrix}
\begin{array}{c|c|c|c}
a_1 & a_2 & \cdots & a_n
\end{array} 
\end{bmatrix} = \begin{bmatrix}
\begin{array}{c|c|c|c}
e_1 & e_2 & \cdots & e_n
\end{array} 
\end{bmatrix}  \begin{bmatrix}
a_1 \cdot e_1 & a_2 \cdot e_1, & \cdots & a_n \cdot e_1 \\
0 & a_2 \cdot e_2 & \cdots & a_n \cdot e_2 \\
\vdots & 
\end{bmatrix}
$$

With the above formula, one could do QR calculation easily by:

- normalize each columns for Q 
- calculate each element for R with dot production 


One can see that this algorithm is not stable. We need to change the order
of operations. The algorithm is then called the __modified Gram-Schmidt__
(MGS) algorithm. 

```python
def modified_gram_schmidt(A: jnp.ndarray, reduced=False) -> np.ndarray:
    """
    Modified gram schmidt 
    """
    m, n = A.shape
    # it is important to use float64
    # the algorithm will not work for intergers without it 
    Q = A.copy()
    R = jnp.zeros_like(A)
    # iterate over columns 
    for i in range(n):
        # calculate the norm of each column 
        # assign it to the diagonal element of R
        R = R.at[i, i].set(jnp.linalg.norm(Q[:, i]))
        # normalize the ith column of Q 
        Q = Q.at[:, i].set(Q[:, i]/R[i, i])
        for j in range(i+1, n):
            # dot production for each element of R
            # fix row i and iterate over columns (upper triangle)
            # check the formula above 
            R = R.at[i, j].set(jnp.dot(Q[:, i], Q[:, j])) # projection coefficient 
            Q = Q.at[:, j].set(Q[:, j] - R[i, j] * Q[:, i]) # construct new axis
    
    if reduced:
        return Q, R[:n, :n]

    return Q, R


def mgs(A: jnp.ndarray):
    """
    different version of modified gram schmidt 
    """
    m, n = A.shape
    R = jnp.zeros((n, n))
    
    for j in range(n):
        for i in range(j-1):
            for k in range(m):
                temp = A[k, i] * A[k, j]
                R = R.at[i, j].set(R[i, j] + temp)
            for k in range(m):
                temp = A[k, i] * R[i, j]
                A = A.at[k, j].set(A[k, j] - temp)
                
        R = R.at[j, j].set(jnp.linalg.norm(A[:, j]))
        A = A.at[:, j].set(A[:, j] / R[j, j])
        
    return A, R
```

__Recap__. We've seen three ways to compute the QR decomposition: Householder
reflection, Givens rotations, and Gram-Schmidt. Householder reflections
are a great approach if $A$ is arbitrary and (close to) square. Gives 
rotations are useful when $A$ is upper Hessenberg. Gram-Schmidt is a 
goode idea when $A$ is tall and thin, and we want a QR decomposition 
where $Q$ is also tall and thin. 

QR factorization via Householder transformation is usually faster and 
more accurate than Gram-Schmidt methods.


## Upper Hessenberg Form


An upper Hessenberg matrix is a square matrix that is nearly upper 
triangular, with zeros below the first subdiagonal. Every $n \times n$ 
matrix $A$ can be written $A = QHQ^T$ where $Q$ is orthonormal and 
$H$, called the Hessenberg form of $A$, is an upper Hessenberg matrix. 
Putting a matrix in upper Hessenberg form is an important first step to 
computing its eigenvalues numerically.


This algorithm also uses Householder transformations. To find orthogonal 
$Q$ and upper Hessenberg $H$ such that 

$$A = QHQ^T,$$

it suffices to find such matrices that satisfy 

$$Q^TAQ = H$$

Thus, the strategy is to multiply $A$ on the left and right by a
series of orthonormal matrices until it is in Hessenberg form.


Using the same $Q_k$ as in the kth step of the Householder algorithm 
introduces $n − k$ zeros in the $k$th column of $A$, but multiplying 
$Q_kA$ on the right by $Q^T_k$ destroys all of those zeros. 

Instead, choose a $Q_1$ that fixes $e_1$ and reflects the first column of $A$ 
into the span of $e_1$ and $e_2$. The product $Q_1A$ then leaves the 
first row of $A$ alone, and the product $(Q_1A)Q^T_1$ leaves the 
first column of $(Q_1A)$ alone.


<div class='figure'>
    <img src="/math/images/householder8.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 9.</span> The illustration of Hessberg form.
    </div>
</div>

Continuing the process results in the upper Hessenberg form of $A$.

<div class='figure'>
    <img src="/math/images/householder9.png"
         alt="floating number illustrated"
         style="width: 50%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 10.</span> The illustration of Hessberg form.
    </div>
</div>


This implies that $A = Q_1^T Q_2^T Q_3^T H Q_3 Q_2 Q_1$, so setting $Q = Q_1^T Q_2^T Q_3^T$
results in the desired factorization $A = QHQ^T$.


```python
def hessenberg(A: jnp.ndarray) -> jnp.ndarray:
    """
    Transform a matrix A into the Hessenberg form via Householder reflection
    It is very similar to the algorithm of QR with Householder 
    """
    m, n = A.shape
    H = A.copy()
    Q = jnp.eye(m)
    
    # construct sign function
    sign = lambda x: 1 if x >= 0 else -1
    # instead iterating all columns we loop over it upto n-2
    for i in range(n-2):
        # starting from the second row 
        v = H[i+1:, i].reshape(-1, 1)
        v_norm = jnp.linalg.norm(v)
        v = v.at[0, 0].set(v[0, 0] + sign(v[0, 0]) * v_norm)
        # normalize v
        v = v.at[:, :].set(v/jnp.linalg.norm(v))
        
        # update H and Q
        # apply Q_k H (H is copy of A)
        H = H.at[i+1:, i:].set(H[i+1:, i:] - 2 * v @ v.T @ H[i+1:, i:])
        # apply Q_k^T to H, H Q_k^T 
        H = H.at[:, i+1:].set(H[:, i+1:] - 2 * (H[:, i+1:] @ v) @ v.T)
        # calculate Q by Q_kQ_{k-1}
        Q = Q.at[i+1:, :].set(Q[i+1:, :] - 2 * v @ v.T @ Q[i+1:, :])
        
    return H, Q.T


def hessenberg_np(A):
    """
    Hessenberg decomposition 
    """
    m, n = A.shape
    # convert to float64 incase inputs are integer
    H = A.copy().astype(np.float64)  
    Q = np.eye(m)
    # construct sign function
    sign = lambda x: 1 if x >= 0 else -1
    # iterate over columns 
    for i in range(n-2):
        # get the column and reshape it into column vector 
        # it is important to copy 
        # as matrix is mutable in numpy 
        x = H[i+1:, i].reshape(-1, 1).copy()
        # calculate the column
        x_norm = np.linalg.norm(x)
        # update the first entry 
        x[0, 0] = x[0, 0] + sign(x[0,0]) * x_norm
        # normalize the vector
        x /= np.linalg.norm(x)

        # update H and Q
        H[i+1:, i:] = H[i+1:, i:] - 2 * x @ x.T @ H[i+1:, i:] 
        H[:, i+1:] = H[:, i+1:] - 2 * (H[:, i+1:] @ x) @ x.T 
        Q[i+1:, :] = Q[i+1:, :] - 2 * x @ x.T @ Q[i+1:, :]

    # transpose Q 
    return H, Q.T
```











All figures in this post are taken from the book by {% cite darve2021numerical %}.

{% endkatexmm %}