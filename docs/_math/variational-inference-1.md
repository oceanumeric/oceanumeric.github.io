---
title: Variational Inference (1)
subtitle: Modern Bayesian statistics relies on models for which the posterior is not easy to compute and corresponding algorithms for approximating them. Variational inference is one of the most popular methods for approximating the posterior. In this post, we will introduce the basic idea of variational inference and its application to a simple example.
layout: math_page_template
date: 2023-04-29
keywords: probabilistic-thinking machine-learning bayesian-inference bayesian-statistics variational-inference
published: true
tags: probability algorithm data-science machine-learning  bayesian-statistics variational-inference
---

I don't know why I got hooked on variational inference. Maybe it's because I'm a Bayesian and I want to know how to do Bayesian inference in practice. Maybe it's because it has many applications in generative models with deep learning. Anyway, I think it's a good idea to write down what I have learned about variational inference. This is the first post of a series of posts about variational inference. In this post, we will introduce the basic idea of variational inference and its application to a simple example.

This series of posts again will take the perspective from information theory as I think it really makes our life easier.


## Big picture

{% katexmm %}

When Shannon introduced the concept of entropy, he was asking the question: how much is our uncertainty reduced by learning an outcome of a random variable? In other words, how much information do we gain by learning an outcome of a random variable? This is the question that information theory tries to answer.

To measure the information gain is to measure uncertainty. To come up an equation that could measure uncertainty, we need to design some criteria first {% cite mcelreath2020statistical %}. The criteria that Shannon used is the following:

1. The measure should be continuous.
2. The measure should be additive.
3. The measure of uncertainty should increase as the number of possible outcomes increases.

The function satisfying the above criteria is the following:

$$
H(X) = -\sum_{i=1}^n p(x_i) \log p(x_i). \tag{1}
$$

This function is called the entropy of a random variable $X$. It measures the uncertainty of a random variable. The more uncertain a random variable is, the higher its entropy is.

Let's have a simple example to illustrate the idea.

| hello | hello_world | hello_word2 | hello_int | hello_int2 | hello_int3 |
|-------|-------------|-------------|-----------|------------|------------|
| hello | hello       | hello       |         1 |          1 |          1 |
| hello | world       | tomorrow    |         1 |          2 |          2 |
| hello | hello       | will        |         1 |          1 |          3 |
| hello | hello       | be          |         1 |          1 |          4 |
| hello | hello       | a           |         1 |          1 |          5 |
| hello | hello       | great       |         1 |          1 |          6 |
| hello | hello       | day         |         1 |          1 |          7 |
| hello | hello       | hello       |         1 |          1 |          1 |
| hello | hello       | hello       |         1 |          1 |          1 |
| hello | hello       | hello       |         1 |          1 |          1 |



The table above shows three columns with different kinds of 'information', and we are using integer to represent each word. This means we could calculate the probability for each variable by setting sample size $n = 10$. By reading those words, one could see that the third column gives more information as it introduces more uncertainty (or variance). This is also reflected by the entropy of each column as it is shown in the figure below.


<div class='figure'>
    <img src="/math/images/entropy.png"
         alt="Entropy illustration"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The plot of entropy for each column. The green one is for the third column.
    </div>
</div>


After having a sense of entropy, we could move on to the next question: how to measure the distance between two probability distributions? This is the question that Kullback and Leibler (KL) divergence tries to answer. The KL divergence is defined as the following:

$$
D_{KL}(p, q) = \sum_{i=1}^n p(x_i) \log \frac{p(x_i)}{q(x_i)} = \sum_{i=1}^n p(x_i) \left [ \log p(x_i) -  \log q(x_i) \right ]. \tag{2}
$$

Equation (2) is the KL divergence between two probability distributions $p$ (the target) and $q$ (the model). This equations is often called the cross-entropy between $p$ and $q$.

With those two concepts, we will use a concrete example to illustrate the idea of variational inference, which is about expectation maximization (EM) algorithm.


## Density estimation

Suppose we have a dataset $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$, and we want to estimate the density of the data. Let's run a simulation to generate some data and see how it looks like.

{% endkatexmm %}

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def figure1():
    # set seed  
    np.random.seed(57)
    # sample size 100
    n = 100
    # sample mean 1 and 10
    mu1, mu2 = 1, 10
    # use same standard deviation 1
    sigma = 1
    # generate two normal distributions
    x1 = np.random.normal(mu1, sigma, n)
    x2 = np.random.normal(mu2, sigma, n)

    # combine two distributions
    x = np.concatenate((x1, x2))


    # plot the distributions
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(x[:n], np.zeros_like(x[:n]),
               alpha=0.5, marker=2, color="green")
    ax.scatter(x[n:], np.zeros_like(x[n:]),
               alpha=0.5, marker=2, color="#6F6CAE")
    _ = ax.set_yticks([])
    sns.histplot(x[:n], color="green", alpha=0.5,
                    kde=True,  ax=ax)
    sns.histplot(x[n:], color="#6F6CAE", alpha=0.5,
                    kde=True, ax=ax)
    ax.set_title("Two normal distributions")
    # add legend
    ax.legend(["$\mathcal{N}(1, 1)$", "$\mathcal{N}(10, 1)$"],
                        frameon=False)
```
<div class='figure'>
    <img src="/math/images/variational-inference-figure-1.png"
         alt="Entropy illustration"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The histogram and density plot of two normal distributions based on the simulation.
    </div>
</div>

{% katexmm %}

with this dataset, we know there are two normal distributions. However, if we don't know the data generating process, we could only estimate the density of the data.

To classify the data, we could introduce a latent variable $z$ to represent the class of the data, $z = (z^{(1)}, \cdots, z^{(m)})$, which is multinomial distributed, and $z^{(i)}$ represents the class of the $i$-th data point. For our example in figure 2, we have two classes, so $z^{(i)} \in \{0, 1\}$, which is a Bernoulli distribution.

Now, we could model the data generating process as the following:

latex symbol for big theta is 

$$
p(x, z; \Theta); \quad \Theta \text{is the parameter (mean, etc.) of the model}. \tag{3}
$$

Since we only observe the data $x$, we could marginalize the latent variable $z$ to get the marginal distribution of $x$:

$$
p(x; \Theta) = \sum_z p(x, z; \Theta). \tag{4}
$$

We could use the maximum likelihood estimation (MLE) to estimate the parameter $\Theta$ by maximizing the log-likelihood of the data, such
as

$$
\begin{aligned}
\Theta^* & = \arg \max_\Theta \prod_{i=1}^n p(x^{(i)}; \Theta) \\
         & = \arg \max_\Theta \sum_{i=1}^n  \ln p(x; \Theta) \\
         & = \arg \max_\Theta \sum_{i=1}^n  \ln \sum_z p(x, z; \Theta).
\end{aligned} \tag{5}
$$

To solve this problem directly, we need to calculate the derivative of the log-likelihood with respect to the parameter $\Theta$, which is not easy to do (we have to calculate the derivative of the log of the sum of the probability because of the latent variable $z$).

Instead, we could use the EM algorithm to solve this problem. To understand the EM algorithm, we need to introduce the concept of lower bound, which is based on Jensen's inequality.


## Jensen's inequality


Jensen's inequality states that for a convex function $f$, we have

$$
f(\mathbb{E}[x]) \leq \mathbb{E}[f(x)]. \tag{6}
$$

If the function $f$ is concave, then we have

$$
f(\mathbb{E}[x]) \geq \mathbb{E}[f(x)]. \tag{7}
$$

Now, let $f(x) = \ln x$, which is a concave function, then we have

$$
\ln \mathbb{E}[x] \geq \mathbb{E}[\ln x]. \tag{8}
$$

## EM algorithm

Now, we could use the Jensen's inequality to derive the EM algorithm. First, we could rewrite the log-likelihood in equation (4) as the following:

$$
\begin{aligned}
\ln p(x; \Theta) & =  \ln \sum_z p(x, z; \Theta) \\
                 & =  \ln \sum_z q(z) \frac{p(x, z; \Theta)}{q(z)} \\
                 & \geq \sum_z q(z) \ln \frac{p(x, z; \Theta)}{q(z)} 
\end{aligned} \tag{9}
$$

where $q(z)$ is a distribution over the latent variable $z$. The last step is based on the Jensen's inequality. Now, we could substitute the lower bound of the log-likelihood into the MLE problem in equation (5):

$$
\begin{aligned}
L(\Theta, q) & =  \sum_{i=1}^n  \ln \sum_z p(x, z; \Theta) \\ 
             & \geq \sum_{i=1}^n \sum_z q(z) \ln \frac{p(x, z; \Theta)}{q(z)} 
\end{aligned} \tag{10}
$$

This is the lower bound of the log-likelihood. Now, we could maximize the lower bound with respect to the parameter $\Theta$ and the distribution $q(z)$, which is equivalent to maximizing the log-likelihood. 


Now, if we fix the supremum of the lower bound, then we could have

$$
\frac{p(x, z; \Theta)}{q(z)} = \text{const} = c \tag{11}
$$

This leads to 

$$
q(z) \propto p(x, z; \Theta); \quad \ s.t. \sum_z q(z) = 1. \tag{12}
$$

This shows that the distribution $q(z)$ is the posterior distribution of the latent variable $z$ given the data $x$ and the parameter $\Theta$. Therefore, we could have

$$
p(z|x; \Theta) = \frac{p(x, z; \Theta)}{p(x; \Theta)} = \frac{p(x, z; \Theta)}{\sum_z p(x, z; \Theta)} = q(z). \tag{13}
$$

This is the E-step of the EM algorithm. In the E-step, we calculate the posterior distribution of the latent variable $z$ given the data $x$ and the parameter $\Theta$.

For the M-step, we maximize the lower bound with respect to the parameter $\Theta$:

$$
\Theta^* = \arg \max_\Theta \sum_{i=1}^n  q(z) \ln \frac{\sum_z p(x, z; \Theta)}{q(z)}  \tag{14}
$$

Before we implement the EM algorithm, we will link the EM algorithm to KL divergence in equation (2).

## Evidence Lower Bound (ELBO)







{% endkatexmm %}













