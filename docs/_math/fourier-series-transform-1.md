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
Informally, the expected value is the weighted arithmetic mean of a large number of independently selected outcomes of a random variable. This means we have:
<i>one random variable</i> and <i> a countable set of possible outcomes</i>. 
For example, weight of human being follows the normal distribution: random 
variable is weight of human being, a countable set of possible outcomes 
is the set of all possible values of weight (some people are fit, some are
slim, some are overweight). 

At last, we need to calculate the <b>weighted</b> arithmetic mean. 
</p>













{% endkatexmm %}