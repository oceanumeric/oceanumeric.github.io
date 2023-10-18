---
title: Fourier Series and Fourier Transform - I
subtitle: Certain feelings in my body lead me to believel that I have to stduy Fourier Series and Fourier Transform for a better understanding of probability theory, measure theory,entroy and information theory.
layout: math_page_template
date: 2023-10-15
keywords: fourier series, fourier transform, probability, measure theory, entropy, foundations of probability
published: true
tags: fourier-analysis probability measure-theory entropy foundations-of-probability
---

For people who are in the field of AI, it is well known that 'neural networks can approximate any function given enough hidden neurons'. This is a very powerful statement. But, what does it mean? What is the proof? what's the connection between this statement and linear algebra space, such as project matrix, eigenvalue, eigenvector, etc.? What's the role of distribution of input plays in this statement? What's the role of activation function plays in this statement? 


To answer these questions, we need to understand Fourier Series and Fourier Transform. I have been trying to master Fourier Series and Fourier Transform for a long time. But,
it takes me sveral years to understand the basic concepts of Fourier Series and Fourier Transform. I am still learning it. I will try to explain what I have learned so far in this series of blog posts.

If you have read my previous post about probability and distribution, you might notice
that sometimes we have to talk about Gamma function, Beta function, etc. These functions are related to Fourier Series and Fourier Transform. So, I will also try to explain these functions in this series of blog posts. The point is that to understand
the basic concepts of probability theory, we have to understand Fourier Series and Fourier Transform. Since probabiliy theory is the foundation of AI and even quantum mechanics, we have to understand Fourier Series and Fourier Transform.

- [Where to start?](#where-to-start)
- [Euler's number and Euler's formula](#eulers-number-and-eulers-formula)
- [Dot product and inner product](#dot-product-and-inner-product)
- [Two ways of deriving least square solution](#two-ways-of-deriving-least-square-solution)
- [Back to Euler's formula](#back-to-eulers-formula)
- [Fourier Series](#fourier-series)


## Where to start?

{% katexmm %}

Both probability theory and fourier analysis (Fourier Series and Fourier Transform) are very deep subjects. They are also very rich subjects. One could easily spend her/his entire life to study them. So, where to start? We need to make a choice, which certainly involves some trade-offs.

The choice I am making is to have an advanced standpoint that will allow us to have
a unified view of fourier analysis, probability theory and their applications in the field of AI. This advanced standpoint is both elementary and advanced. It is elementary in the sense that it does not require any advanced math background. It is advanced in the sense that it will lead us to mroe advanced topics in the future.

The standpoint I am taking is to start with the following concepts:

- theoretical mathematics v.s. applied mathematics
- idealized v.s. real world
- continuous v.s. discrete

To explain this standpoint, let's take a look at a specific example. Look at the following piciture:

<div class='figure'>
    <img src="/math/images/Isosceles_right_triangle.svg"
         alt="Inequality bounds compare"
         style="width: 30%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The right triangle with sides of length $a=b=1$ has hypotenuse of length $\sqrt{2}$.
    </div>
</div>

Now, you want to estimate the value of $\sqrt{2}$. People from different regions have
different ways to estimate the value of $\sqrt{2}$. Please read [this](https://en.wikipedia.org/wiki/Square_root_of_2) if you are interested in the history of $\sqrt{2}$.

Now living in the 21st centry, how could we estimate the value of $\sqrt{2}$. We can rely on Taylor series to estimate the value of $\sqrt{2}$. Let's review Taylor series first.

### Taylor series

Talyor series is a way to approximate a function $f(x)$ by a polynomial $p(x)$. Here is the formula:

$$
\begin{aligned}
p(x) & = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x-a)^n \\
     & = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots
\end{aligned}
$$

where $f^{(n)}(a)$ is the $n$-th derivative of $f(x)$ evaluated at $x=a$. Now, we let
$f(x) = \sqrt{x} = x^{1/2}$ and $a=1$. We know

$$
\begin{aligned}
f(x) & = x^{1/2} \quad \Rightarrow \quad f(1) = 1 \\
f'(x) & = \frac{1}{2} x^{-1/2}  \quad \Rightarrow \quad f'(1) = \frac{1}{2} \\
f''(x) & = -\frac{1}{4} x^{-3/2}  \quad \Rightarrow \quad f''(1) = -\frac{1}{4} \\
f'''(x) & = \frac{3}{8} x^{-5/2}  \quad \Rightarrow \quad f'''(1) = \frac{3}{8} \\
f^{(4)}(x) & = -\frac{15}{16} x^{-7/2}  \quad \Rightarrow \quad f^{(4)}(1) = -\frac{15}{16} \\
\end{aligned}
$$

Then, we have

$$
\begin{aligned}
p(x) & = \sum_{n=0}^{\infty} \frac{f^{(n)}(1)}{n!} (x-1)^n \\
     & = f(1) + f'(1)(x-1) + \frac{f''(1)}{2!}(x-1)^2 + \frac{f'''(1)}{3!}(x-1)^3 + \cdots \\
     & = 1 + \frac{1}{2}(x-1) - \frac{1}{8}(x-1)^2 + \frac{1}{16}(x-1)^3 + \cdots
\end{aligned}
$$

If we let $x=2$, we have

$$
\begin{aligned}
p(2) & = 1 + \frac{1}{2}(2-1) - \frac{1}{8}(2-1)^2 + \frac{1}{16}(2-1)^3 + \cdots \\
     & = 1 + \frac{1}{2} - \frac{1}{8} + \frac{1}{16} + \cdots \\
     & = ?  \quad \text{(let's write a program to compute this value)}
\end{aligned}
$$


<div class='figure'>
    <img src="/math/images/taylor_series_estimation.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The Taylor series estimation of $\sqrt{2}$, and its convergence.
    </div>
</div>


```python
#%%
import numpy as np
import matplotlib.pyplot as plt


# f(x) = x^{1/2}
def f(x, power=0.5):
    return x**power

# derivative of f(x)
def df(x, n):
    # not a general solution but works for this case
    # assume n >= 1
    # using recursion
    if n == 1:
        return 1/2 * x**(-1/2)
    else:
        return (1/2 - (n-1)) * x**(1/2-n) * df(x, n-1)


def taylor_series(x, n, a=1):
    # n is integer assume n >= 1
    ts = 0
    for i in range(1, n):
        ts += df(a, i) * (x-a)**i / np.math.factorial(i)
    return ts + f(a)


# plot taylor series
def plot_taylor_series():
    n = 17  # set up to 16th order
    nlist = list(range(1, 10))
    y = [taylor_series(2, i) for i in nlist]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(nlist, y, 'k--', label='Taylor series')
    # add np.sqrt(2) as a horizontal line
    ax.axhline(y=np.sqrt(2), color='k', linestyle='-', label='sqrt(2)')
    ax.set_xlabel('order')
    ax.set_ylabel('value')
    ax.set_title('Taylor series of sqrt(2) at x=2 around a=1')
    # legend right bottom
    ax.legend(loc='lower right')
        


if __name__ == "__main__":
    # plot option retinal display
    plt.rcParams['figure.dpi'] = 300
    print("hello world")
    print(f(1))
    print(df(1, 4))
    print(taylor_series(2, 6))
    # retinal display
    %config InlineBackend.figure_format = 'retina'
    plot_taylor_series()
```

Figure 2 shows the Taylor series estimation of $\sqrt{2}$, and its convergence. We can see that the Taylor series estimation of $\sqrt{2}$ converges to $\sqrt{2}$ as the order of the Taylor series increases. The Taylor series estimation of $\sqrt{2}$ is a good estimation of $\sqrt{2}$. 

<p class='theorembox'>
<b>Reflections</b>
<br>
This simple example enables us to reflect on the many topics: i) the meaning of
theoretical mathematics v.s. applied mathematics, ii) the meaning of idealized v.s. real world, iii) the meaning of continuous v.s. discrete. iv) the connection between analysis and linear algebra, etc. 
<br>
<b>The most important message</b> I want to convey is that one has to develop the 
certain level of mathematical maturity to carry out the mathematical analysis and 
implement the calculation in the real world, which is so called 'engineering'.
</p>

When we use the Taylor series to estimate $\sqrt{2}$, we did not cover the following interesting and important questions:

- how fast does the Taylor series converge to $\sqrt{2}$?
- does the anchor point $a$ matter?
- what's the relationship between the Taylor series and Newton's method?
- why does the Taylor series converge to $\sqrt{2}$?
- what's the relationship between the Taylor series and the linear algebra space in terms of basis, eigenvalue, eigenvector, etc.?
- How could we use the knowledge about Taylor series to guide us to study Neural Networks?
- Why learning those advanced math topics could help us to understand the optimization algorithms, such as gradient descent, etc.?

All those questions will be answered in the future blog posts when we study Fourier Series and Fourier Transform systematically. So, without further ado, let's start our journey of Fourier Series and Fourier Transform.


## Euler's number and Euler's formula

Euler's number is a very important number in mathematics. It is denoted by $e$. It is defined as the following infinite series:

$$
e = \sum_{n=0}^{\infty} \frac{1}{n!} = 1 + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!} + \cdots
$$

where $n!$ is the factorial of $n$. For example, $3! = 3 \times 2 \times 1 = 6$. One cannot overstate the importance of Euler's number. It is almost everywhere in mathematics. For instance, the famous Gaussian distribution is defined as the following:

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

where $\exp(x) = e^x$. In number theroy, probability theory, statistics, etc., Euler's number is everywhere. To learn how Euler discovered this number, please read [this](https://digitalcommons.ursinus.edu/triumphs_analysis/3/). I have not read this note, but put it here for future reference.

With appreciation of Euler's number, we can define Euler's formula as the following:

$$
e^{ix} = \cos(x) + i \sin(x) \tag{1}
$$

where $i$ is the imaginary unit, which is defined as the following:

$$
i^2 = -1
$$

This simple formula connects the exponential function with the trigonometric functions. It is a very powerful formula and also the beautiful one. Please read the [wikipedia](https://en.wikipedia.org/wiki/Euler%27s_formula) to learn more about Euler's formula.

With Euler's formula, we can derive the following two important formulas:

$$
\begin{aligned}
\cos(x) & = \frac{e^{ix} + e^{-ix}}{2} \\
\sin(x) & = \frac{e^{ix} - e^{-ix}}{2i} \tag{2}
\end{aligned}
$$

In our posts, we will use the above two formulas frequently. So, please remember them.

## Dot product and inner product

In linear algebra, we have the concept of dot product and inner product. They are very important concepts. In this section, we will review them. It is important to know that
inner product is a generalization of dot product. So, we will start with dot product first.

The dot product of two vectors $\mathbf{a}$ and $\mathbf{b}$ is defined as the following:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n
$$

where $n$ is the dimension of the vectors $\mathbf{a}$ and $\mathbf{b}$. For example, if $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [4, 5, 6]$, then

$$
\mathbf{a} \cdot \mathbf{b} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32
$$

The dot product of two vectors $\mathbf{a}$ and $\mathbf{b}$ is also called the inner product of two vectors $\mathbf{a}$ and $\mathbf{b}$. The inner product of two vectors $\mathbf{a}$ and $\mathbf{b}$ is denoted by $\langle \mathbf{a}, \mathbf{b} \rangle$. So, we have

$$
\langle \mathbf{a}, \mathbf{b} \rangle = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n \tag{3}
$$

The inner product has a very nice geometric interpretation:

$$
\langle \mathbf{a}, \mathbf{b} \rangle = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta  \tag{4}
$$

where $\theta$ is the angle between the two vectors $\mathbf{a}$ and $\mathbf{b}$, and $\|\mathbf{a}\|$ and $\|\mathbf{b}\|$ are the lengths of the vectors $\mathbf{a}$ and $\mathbf{b}$, respectively. We also refer to $\|\mathbf{a}\|$ as the norm of the vector $\mathbf{a}$. The norm of a vector $\mathbf{a}$ is defined as the following:

$$
\|\mathbf{a}\| = \sqrt{\langle \mathbf{a}, \mathbf{a} \rangle} = \sqrt{\mathbf{a} \cdot \mathbf{a}} = \sqrt{\sum_{i=1}^{n} a_i^2}  \tag{5}
$$

Before we move on, we need to define the inner product for complex functions. Suppose we have two complex functions $f(t)$ and $g(t)$, where $t$ is a real number. The inner product of $f(t)$ and $g(t)$ is defined as the following:

$$
\langle f(t), g(t) \rangle = \int_{0}^{1} f(t) \overline{g(t)} dt
$$

where $\overline{g(t)}$ is the complex conjugate of $g(t)$. For example, if $g(t) = e^{2\pi int}$, then $\overline{g(t)} = e^{-2\pi int}$. The inner product of $f(t)$ and $g(t)$ is also called the dot product of $f(t)$ and $g(t)$.

Please read [this](https://en.wikipedia.org/wiki/Complex_conjugate) to learn more about the complex conjugate.



## Two ways of deriving least square solution

In this section, we will review two ways of deriving the least square solution. The first way is to use the calculus. The second way is to use the linear algebra. Again,
the purpose of doing this is to give you certain level of mathematical maturity, especailly on the relationship between analysis and algebra.

Suppose the regression model has the following matrix model

$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix} = \begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np} \\
\end{bmatrix} \begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_p
\end{bmatrix} + \begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_n
\end{bmatrix} 
$$

where $y_i$ is the $i$-th observation of the dependent variable, $x_{ij}$ is the $j$-th observation of the $i$-th independent variable, $\beta_j$ is the coefficient of the $j$-th independent variable, and $\epsilon_i$ is the error term of the $i$-th observation. We assume that the error term $\epsilon_i$ is independent and identically distributed (i.i.d.) with mean zero and variance $\sigma^2$. We also assume that the error term $\epsilon_i$ is independent of the independent variables $x_{ij}$. 

The equation (6) could be written in the following matrix form:

$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon} 
$$

where $\mathbf{y}$ is the vector of the dependent variable, $\mathbf{X}$ is the matrix of the independent variables, $\boldsymbol{\beta}$ is the vector of the coefficients, and $\boldsymbol{\epsilon}$ is the vector of the error terms. 

Now, our goal is to find the estimation of the coefficients $\boldsymbol{\beta}$, denoted by $\hat{\boldsymbol{\beta}}$ by minimizing the residual sum of squares (RSS), which is defined as the following:

$$
RSS = \sum_{n=1}^N \hat{\epsilon}_i^2 = \sum_{n=1}^N (y_i - \hat{y}_i)^2 = \sum_{n=1}^N (y_i - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2 
$$

where $\hat{\epsilon}_i$ is the residual of the $i$-th observation, $\hat{y}_i$ is the predicted value of the $i$-th observation, and $\mathbf{x}_i$ is the vector of the $i$-th observation of the independent variables.

Since $\hat{\epsilon} = y - X \hat{\boldsymbol{\beta}}$, we have

$$
\begin{aligned}
RSS & = \hat{\epsilon}^T \hat{\epsilon} = (y - X \hat{\boldsymbol{\beta}})^T (y - X \hat{\boldsymbol{\beta}}) \\
& = y^T y - 2y^T X \hat{\boldsymbol{\beta}} + \hat{\boldsymbol{\beta}}^T X^T X \hat{\boldsymbol{\beta}} \\
\end{aligned}
$$

The first order condition of the above equation is the following:

$$
\begin{aligned}
\frac{\partial RSS}{\partial \hat{\boldsymbol{\beta}}} & = -2 X^T y + 2 X^T X \hat{\boldsymbol{\beta}} = 0 \\
\Rightarrow \quad \hat{\boldsymbol{\beta}} & = (X^T X)^{-1} X^T y
\end{aligned} \tag{6}
$$

This is the least square solution. We can also derive the least square solution by using the linear algebra. To do this, we need to reply on the following two facts:

- the projection of a vector $\mathbf{y}$ onto a vector space $\mathcal{C}(\mathbf{X})$ is the orthogonal projection of $\mathbf{y}$ onto $\mathcal{C}(\mathbf{X})$.
- the difference between $\mathbf{y}$ and its projection onto $\mathcal{C}(\mathbf{X})$ is orthogonal to $\mathcal{C}(\mathbf{X})$.

Therefore, we have

$$
X^T (y - X \hat{\boldsymbol{\beta}}) = 0 
$$

where $X\hat{\boldsymbol{\beta}}$ is the projection of $\mathbf{y}$ onto $\mathcal{C}(\mathbf{X})$. This will lead to the same least square solution as the one derived by using the calculus.


## Back to Euler's formula

I hope at this stage, you will have developed an intuition that:

- the calculus is about the analysis
- the linear algebra is about the algebra and calculation

In the real world, we need both analysis and algebra to solve the real world problems,
especially for the problems in the field of probability and AI. 

Now, let's study the Euler's formula again. We have

$$
e^{ix} = \cos(x) + i \sin(x) 
$$

Now, we will extend the above formula to the complex exponentials. To connect the
analysis and algebra, we will rely on the following series:

- the complex expoenentials $e^{2\pi int}$, where $n = 0, \pm 1, \pm 2, \cdots$ and $t$ is a real number. We write

$$
e_n(t) = e^{2\pi int}
$$

The inner product of two of them, $e_n(t)$ and $e_m(t)$ is the following:

$$
\begin{aligned}
\langle e_n(t), e_m(t) \rangle = \int_{0}^{1} e^{2\pi int} e^{-2\pi imt} dt & = \int_{0}^{1} e^{2\pi i(n-m)t} dt  \\
& = \begin{cases}
1, & \text{if } n = m \\
0, & \text{if } n \neq m
\end{cases}
\end{aligned} \tag{7}
$$

When $n=m$, the derivation is easy. When $n \neq m$, we have

$$
\begin{aligned}
\int_{0}^{1} e^{2\pi i(n-m)t} dt & = \frac{1}{2\pi i(n-m)} e^{2\pi i(n-m)t} \bigg|_{0}^{1} \\
& = \frac{1}{2\pi i(n-m)} (e^{2\pi i(n-m)} - 1) \\
& = \frac{1}{2\pi i(n-m)} (1 - 1) \\
& = 0
\end{aligned}
$$

$$
e^{2\pi i(n-m)} = \cos(2\pi(n-m)) + i \sin(2\pi(n-m)) = 1
$$

Therefore, the complex exponentials $e_n(t)$ are orthogonal to each other. They form
an orthonormal basis of the vector space $\mathcal{C}([0, 1])$, which is quite similar to the standard basis of the vector space $\mathbb{R}^n$.


<p class='theorembox'>
<b>Reflections</b>
I hope you could apprecite $e_n(t)$ and $e_m(t)$ are orthogonal to each other. This is the key to understand the Fourier Series and Fourier Transform.
<br>
What I like about the above derivation is that it connects the analysis and algebra. The analysis is about the inner product in complex field, and the algebra is about the orthonormal basis. Again, let's look at it $e_n(t) = e^{2\pi int}$, where $n = 0, \pm 1, \pm 2, \cdots$ and $t$ is a real number. You have real number $t$,  complex number $e^{2\pi int}$, and a sequence with index $n$. This is a very rich structure.
</p>


## Fourier Series

Now, we are ready to study the Fourier Series. The Fourier Series starts with the following question:

- Given a function $f(t)$, how could we represent it as a linear combination of the complex exponentials $e_n(t)$?

To answer this question, we need to find the coefficients of the linear combination. To find the coefficients, we need to rely on the inner product.


Now, suppose we could write $f(t)$ as the following:

$$
f(t) = \sum_{n=-N}^{N} c_n e_n(t) = \sum_{n=-N}^N c_n e^{2\pi int} \tag{8}
$$

Now, we will fix an index $k$, and pull the $k$-th term out of the summation. We have

$$
c_k e^{2\pi ikt} = f(t) - \sum_{n=-N, n \neq k}^N c_n e^{2\pi int} 
$$

Now, multiply both sides by $e^{-2\pi ikt}$, and integrate both sides from $0$ to $1$. We have

$$
\begin{aligned}
\int_{0}^{1} c_k e^{2\pi ikt} e^{-2\pi ikt} dt & = \int_{0}^{1} f(t) e^{-2\pi ikt} dt - \int_{0}^{1} \sum_{n=-N, n \neq k}^N c_n e^{2\pi int} e^{-2\pi ikt} dt \\
& = \int_{0}^{1} f(t) e^{-2\pi ikt} dt - \sum_{n=-N, n \neq k}^N c_n \int_{0}^{1} e^{2\pi i(n-k)t} dt \\
& = \int_{0}^{1} f(t) e^{-2\pi ikt} dt - \sum_{n=-N, n \neq k}^N c_n \delta_{nk} \\
\end{aligned}
$$

where $\delta_{nk}$ is the Kronecker delta, which is orthonormal. Therefore, we have

$$
c_k = \int_{0}^{1} f(t) e^{-2\pi ikt} dt = \langle f(t), e_k(t) \rangle \tag{9}
$$

Therefore the fourier series coefficients $c_k$ are the inner product of the function $f(t)$ and the complex exponentials $e_k(t)$. Or you can say that the fourier series coefficients $c_k$ are the projection of the function $f(t)$ onto the vector space $\mathcal{C}([0, 1])$ spanned by the complex exponentials $e_k(t)$.

<p class='theorembox'>
<b>Reflections</b>
We keep repeating the same idea: the analysis (calculus and integration) and the algebra (linear algebra and inner product). Hope you now could have a unified view of the analysis and algebra.
</p>



{% endkatexmm %}