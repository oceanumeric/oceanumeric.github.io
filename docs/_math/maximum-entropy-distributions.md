---
title: Maximum Entropy Distributions
subtitle: The connection between entropy and probability distributions is really interesting. In this post, I will explore the connection between entropy and probability distributions, and how we can use this connection to derive the most likely probability distribution given some constraints.
layout: math_page_template
date: 2023-05-06 
keywords: probabilistic-thinking machine-learning bayesian-inference bayesian-statistics maximum-entropy-distribution
published: true
tags: probability entropy machine-learning bayesian-statistics maximum-entropy-distribution
---

When I was studying mathematics in university, one of my professors asked me why I
switched from economics to mathematics. I told him that I was hooked by the normal distribution. To me, the phenomenon that the normal distribution appears everywhere in nature is really fascinating. I want to understand why it is so common. There might be some deep reason behind it. Because I want to find the secret of nature, I fall in love with mathematics.

To understand why the normal distribution is so common, we could look at it from the perspective of convolution. The normal distribution is the convolution of many independent random variables. The central limit theorem tells us that the convolution of many independent random variables will converge to the normal distribution. This is why the normal distribution is so common.

However, there is another way to look at it. The normal distribution is the probability distribution that has the maximum entropy. In this post, I will explore the connection between entropy and probability distributions, and how we can use this connection to derive the most likely probability distribution given some constraints.

- [Entropy](#entropy)
- [Maximum entropy distribution](#maximum-entropy-distribution)
- [Maximum entropy distribution with mean and variance constraints](#maximum-entropy-distribution-with-mean-and-variance-constraints)


## Entropy

{% katexmm %}

Entropy is a measure of uncertainty. The higher the entropy, the more uncertain we are. The entropy of a discrete random variable $X$ is defined as

$$
H(X) = - \sum_{x \in \mathcal{X}} p(x) \log p(x) \tag{1}
$$

where $\mathcal{X}$ is the set of all possible values of $X$. The entropy of a continuous random variable $X$ is defined as

$$
H(X) = - \int_{x \in \mathcal{X}} p(x) \log p(x) \tag{2}
$$

where $\mathcal{X}$ is the set of all possible values of $X$.


## Maximum entropy distribution

Suppose we have a random variable $X$ with a probability distribution $p(x)$. We want to find the probability distribution $p(x)$ that has the maximum entropy. In other words, we want to find the probability distribution $p(x)$ that maximizes the entropy $H(X)$. We will use natural logarithm in this post from now on.

$$
\max_{p(x)} H(X) = - \int_a^b p(x) \ln p(x) dx \tag{3}
$$

We can solve this problem using the method of Lagrange multipliers. We introduce a Lagrange multiplier $\lambda$ and solve the following problem:

$$
\max_{p(x)} H(X) - \lambda \left( \int_{x \in \mathcal{X}} p(x) - 1 \right) \tag{4}
$$

We can solve this problem by taking the derivative of the objective function with respect to $p(x)$ and set it to zero:

$$
\frac{\partial}{\partial p(x)} \left[ - \int_a^b p(x) \ln p(x) dx + \lambda \left( \int_{x \in \mathcal{X}} p(x) - 1 \right) \right] = 0 \tag{5}
$$

We can solve this equation and get

$$
-1 - \ln p(x) + \lambda = 0  \Rightarrow  p(x) = e^{ \lambda - 1} \tag{6}
$$

We can use the constraint $\int_{x \in \mathcal{X}} p(x) - 1 = 0$ to solve for $\lambda$:

$$
\int_{x \in \mathcal{X}} p(x) - 1 =  0  \Rightarrow \int_{x \in \mathcal{X}} e^{\lambda -1} dx = 1  \tag{7}
$$

We can solve this equation and get

$$
\begin{aligned}
\int_{x \in \mathcal{X}} e^{\lambda -1} dx = 1 &\Rightarrow \int_{x \in \mathcal{X}} e^{\lambda -1} dx = e^{\lambda -1} \int_{x \in \mathcal{X}} dx = 1 \\
&\Rightarrow e^{\lambda -1} = \frac{1}{\int_{x \in \mathcal{X}} dx} = \frac{1}{b - a} \\
&\Rightarrow \lambda = \ln (\frac{1}{b - a}) + 1
\end{aligned}
$$

We can plug $\lambda$ back into equation (6) and get

$$
p(x) = \frac{1}{b - a} \tag{8}
$$

This is the uniform distribution. The uniform distribution is the probability distribution that has the maximum entropy when we only have one constraint that $p(x)$ is a probability distribution.

This should not be too unexpected. It is quite intuitive that a uniform distribution is the maximal ignorance distribution (when no other constraints were made). If we know nothing about the distribution of $X$, then the uniform distribution is the best guess we can make.

## Maximum entropy distribution with mean and variance constraints

For a random variable $X$ with a probability distribution $p(x)$, we can define the mean and variance as follows:

$$
\begin{aligned}
\mathbb{E}[X] &= \int_{x \in \mathcal{X}} x p(x) dx  = \mu \\
\mathbb{V}[X] &= \int_{x \in \mathcal{X}} (x - \mu)^2 p(x) dx
\end{aligned} \tag{9}
$$

Now we will solve the following problem:

$$
\begin{aligned}
\max_{p(x)} H(X) + \lambda_1 \left ( \int_{x \in \mathcal{X}} p(x) - 1 \right) & + \lambda_2 \left( \int_{x \in \mathcal{X}} x p(x) - \mu \right) \\
 & + \lambda_3 \left( \int_{x \in \mathcal{X}} (x - \mathbb{E}[X])^2 p(x) - \sigma^2 \right) 
\end{aligned}
\tag{10}
$$

Since $\mu$ is included in the variance, we only need to solve for $\lambda_1$ and $\lambda_3$. Therefore, we could solve the following problem:

$$
\mathcal{L} =  \max_{p(x)} H(X) + \lambda_1 \left( \int_{x \in \mathcal{X}} p(x) dx - 1 \right) + \lambda_3 \left( \int_{x \in \mathcal{X}} (x - \mu)^2 p(x) dx - \sigma^2 \right) \tag{11}
$$

Now, we will take the derivative of the objective function with respect to $p(x)$ and set it to zero:

$$
\begin{aligned}
\frac{\partial}{\partial p(x)}  \bigg [ - \int_a^b p(x) \ln p(x) dx & + \lambda_1 \left( \int_{x \in \mathcal{X}} p(x) - 1 \right) \\
 & + \lambda_3 \left( \int_{x \in \mathcal{X}} (x - \mu)^2 p(x) - \sigma^2 \right) \bigg ] = 0 
\end{aligned} \tag{12}
$$

This  gives us 

$$
-1 - \ln p(x) + \lambda_1 + \lambda_3 (x - \mu)^2 = 0 \Rightarrow p(x) = e^{ \lambda_1 - 1 + \lambda_3 (x - \mu)^2} \tag{13}
$$

Now, we will use the constraint $\int_{x \in \mathcal{X}} p(x) - 1 = 0$ and variance constraint $\int_{x \in \mathcal{X}} (x - \mu)^2 p(x) - \sigma^2 = 0$ to solve for $\lambda_1$ and $\lambda_3$:

$$
\begin{aligned}
\int_{x \in \mathcal{X}} p(x) - 1 =  0 &\Rightarrow \int_{x \in \mathcal{X}} e^{\lambda_1 - 1 + \lambda_3 (x - \mu)^2} dx = 1  \\
\int_{x \in \mathcal{X}} (x - \mu)^2 p(x) - \sigma^2 = 0 &\Rightarrow \int_{x \in \mathcal{X}} (x - \mu)^2 e^{\lambda_1 - 1 + \lambda_3 (x - \mu)^2} dx = \sigma^2
\end{aligned} \tag{14}
$$

For the first constraint, we can solve it and get

$$
\begin{aligned}
\int_{x \in \mathcal{X}} e^{\lambda_1 - 1 + \lambda_3 (x - \mu)^2} dx = 1 &\Rightarrow \int_{x \in \mathcal{X}} e^{\lambda_1 - 1 + \lambda_3 (x - \mu)^2} dx \\
&  = e^{\lambda_1 - 1} \int_{x \in \mathcal{X}} e^{\lambda_3 (x - \mu)^2} dx = 1 \\
&= e^{\lambda_1 - 1} \sqrt{\frac{-\pi}{\lambda_3}} = 1 \\
&\Rightarrow e^{\lambda_1 -1} = \sqrt{\frac{-\lambda_3}{\pi}}
\end{aligned} \tag{15}
$$

In the above derivation, we used the following identity:

$$
\int_{-\infty}^{\infty} e^{-ax^2} dx = \sqrt{\frac{\pi}{a}} \tag{16}
$$

Now, for the second constraint, we can solve it and get

$$
\begin{aligned}
& \int_{x \in \mathcal{X}} (x - \mu)^2 e^{\lambda_1 - 1 + \lambda_3 (x - \mu)^2} dx = \sigma^2 \\ 
&\Rightarrow e^{\lambda_1 - 1} \int_{x \in \mathcal{X}} (x - \mu)^2 e^{\lambda_3 (x - \mu)^2} dx = \sigma^2 
\end{aligned} \tag{17}
$$

To integrate the above equation, we need to use integration by substitution. Let $z =(x - \mu)$. Then $dz = dx$. Therefore, we have

$$
\begin{aligned}
& \int_{x \in \mathcal{X}} (x - \mu)^2 e^{\lambda_1 - 1 + \lambda_3 (x - \mu)^2} dx = \sigma^2 \\
&\Rightarrow e^{\lambda_1 - 1} \int_{z \in \mathcal{X}} z^2 e^{\lambda_3 z^2} dz = \sigma^2 \\
& \Rightarrow e^{\lambda_1 - 1} \sqrt{\frac{\pi}{\lambda_3^3}} = 2 \sigma^2 \\
\end{aligned} \tag{18}
$$

The above integration uses Gamma function to solve the above integration in equation (18). The Gamma function is defined as

$$
\Gamma(z) = \int_0^{\infty} x^{z-1} e^{-x} dx, \quad \mathcal{R}(z) > 0 \tag{19}
$$

For $\int_{z \in \mathcal{X}} z^2 e^{\lambda_3 z^2} dz$, and let $u = \lambda_3 z^2$. Then $du = 2 \lambda_3 z dz$ and $z = \sqrt{\frac{u}{\lambda_3}}$. Therefore, we have

$$
\begin{aligned}
\int_{-\infty}^\infty z^2 e^{\lambda_3 z^2} dz &= \int_{0}^\infty \frac{u}{\lambda_3} e^u \frac{du}{2\lambda_3 z}  \quad \text{($u \geq 0$)} \\ 
& = \int_{0}^\infty \frac{u}{\lambda_3} \sqrt{\frac{\lambda_3}{u}} e^u \frac{du}{2\lambda_3 } \\
& = \frac{1}{2\sqrt{\lambda_3^3}} \int_{0}^\infty \sqrt{u} e^u du \\
& = \frac{1}{2\sqrt{\lambda_3^3}}  \int_{0}^\infty u^{-1/2} e^u du \\
& = \frac{1}{\sqrt{\lambda_3^3}} \Gamma(\frac{1}{2}) \\
& = \frac{1}{2} \sqrt{\frac{\pi}{\lambda_3^3}}
\end{aligned} \tag{20}
$$

We used the special case of the Gamma function $\Gamma(\frac{1}{2}) = \sqrt{\pi}$ in the above derivation.

Now, let's list $\lambda_1$ and $\lambda_3$ in terms of $\sigma^2$:

$$
\begin{aligned}
e^{\lambda_1 -1} & = \sqrt{\frac{-\lambda_3}{\pi}} \\
e^{\lambda_1 - 1} \sqrt{\frac{\pi}{\lambda_3^3}} &= 2 \sigma^2 \\
\end{aligned} \tag{21}
$$

This means we have 

$$
e^{\lambda_1 - 1} = \sqrt{\frac{-\lambda_3}{\pi}}  = 2\sigma^2 \sqrt{\frac{\lambda_3^3}{\pi}} \Rightarrow \lambda_3 = - \frac{1}{2 \sigma^2} \tag{22}
$$

In equation (13), we have shown 

$$
\begin{aligned}
p(x) & = e^{ \lambda_1 - 1 + \lambda_3 (x - \mu)^2} \\
     & = e^{ \lambda_1 - 1} e^{- \frac{1}{2 \sigma^2} (x - \mu)^2} \\
     & = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{- \frac{1}{2 \sigma^2} (x - \mu)^2} \\
     & = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{1}{2 \sigma^2} (x - \mu)^2 \right) \\
     & = \mathcal{N}(\mu, \sigma^2)
\end{aligned}  \tag{23}
$$

This means the probability distribution that has the maximum entropy when we have mean and variance constraints is the normal distribution. This means if a process is random, and with fixed mean and variance, only the normal distribution has the maximum entropy or gives the most information about the process.


## References

For more information about the maximum entropy distribution, please refer to the following references:

1. [Joshua's post](https://joshuagoings.com/2021/06/21/maximum-entropy-distributions/#gaussian-distribution){:target="_blank"}
2. [Sam's post](https://sgfin.github.io/2017/03/16/Deriving-probability-distributions-using-the-Principle-of-Maximum-Entropy/#introduction){:target="_blank"}
3. [Michael-Franke's post](https://michael-franke.github.io/intro-data-analysis/the-maximum-entropy-principle.html){:target="_blank"}
4. [Bjlkeng's post](https://bjlkeng.github.io/posts/maximum-entropy-distributions/){:target="_blank"}

I used ChatGPT to do some derivations in this post. But I found it only works well if you know how to guide it and correct it when it makes mistakes.



{% endkatexmm %}


