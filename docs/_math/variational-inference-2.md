---
title: Variational Inference (2)
subtitle: Modern Bayesian statistics relies on models for which the posterior is not easy to compute and corresponding algorithms for approximating them. Variational inference is one of the most popular methods for approximating the posterior. In this post, we will introduce the basic idea of variational inference and its application to a simple example.
layout: math_page_template
date: 2023-05-01
keywords: probabilistic-thinking machine-learning bayesian-inference bayesian-statistics variational-inference
published: true
tags: probability algorithm data-science machine-learning  bayesian-statistics variational-inference
---

In previous post, we used a simple example - expectation maximization (EM) algorithm to illustrate the basic idea of Bayesian inference. In this post, we will continue to study Bayesian inference, but this time we will focus on a more general and powerful method - variational inference.

## Big picture again

{% katexmm %}

Let's review Bayesian formula first:

$$
\begin{aligned}
p(\theta | x) &= \frac{p(x | \theta) p(\theta)}{p(x)} \\
&= \frac{p(x | \theta) p(\theta)}{\int p(x | \theta) p(\theta) d\theta} \\
\text{posterior} &= \frac{\text{likelihood} \times \text{prior}}{\text{marginal likelihood}} \\ 
& \propto \text{likelihood} \times \text{prior}
\end{aligned} \tag{1}
$$

where $p(\theta | x)$ is the posterior distribution, $p(x | \theta)$ is the likelihood, $p(\theta)$ is the prior distribution, and $p(x)$ is the marginal likelihood.


Usually, we will set up a prior distribution first. With the prior distribution, we can have the value of $p(\theta)$, and then we can calculate the posterior distribution $p(\theta | x)$. However, when we update the posterior distribution, we need to calculate the marginal likelihood $p(x)$, which is the denominator of the formula.


For the continuous case, the marginal likelihood is:

$$
p(x) = \int p(x | \theta) p(\theta) d\theta
$$

For the discrete case, the marginal likelihood is:

$$
p(x) = \sum_{\theta} p(x | \theta) p(\theta)
$$


However, when we have many hidden variables, the marginal likelihood is hard to calculate and the exact calculation sometimes is prohibitively expensive. To solve this problem, we need to resort to approximation schemes, and these fall broadly into two categories, according to whether they rely on _stochastic or deterministic_ approximations: _Markov Chain Monte Carlo_ (MCMC) and _variational inference_ (VI).

Since sampling methods can be computationally expensive, especially for the large data sets that are increasingly common in modern applications. In contrast, variational inference can scale well to large data sets and high-dimensional models.


## Bayesian mixture of Gaussians

Based on the paper by Blei et al. {% cite blei2017variational %}, we will use a Bayesian mixture of Gaussians to illustrate the basic idea of variational inference. The mixture of Gaussians is a simple model that can be used for clustering and density estimation. 

Consider a Bayesian mixture of unit-variance univariate Gaussians (we use univariate Gaussians for simplicity, but the model can be easily extended to multivariate Gaussians). The model has $K$ components, each with a mean $\mu_k$, such as $\mu = \{\mu_1, \mu_2, ..., \mu_K\}$. The mean parameter is drawn independently from a common prior $p(\mu_k)$, which we assume to be a Gaussian 
$N(0, \sigma^2)$, the prior variance $\sigma^2$ is a hyper-parameter. 

Here is the data generating process:

- For each data point $x_i$ in $N$ data points:
    - Sample a cluster assignment $z_i$ uniformly: $z_i \sim \text{Uniform}(K)$
    - Sample a data point $x_i$ from a Gaussian with mean $\mu_{z_i}$: $x_i \sim N(\mu_{z_i}, 1)$
        - where $\mu_{z_i}$ is drawn from a Gaussian with mean $0$ and variance $\sigma^2$: $\mu_{z_i} \sim N(0, \sigma^2)$ with $p(\mu_{z_i})$ being the prior distribution of $\mu_{z_i}$.

We can write down the joint distribution of the model:

$$
\begin{aligned}
p(x, \mu, z) & = p(x | \mu, z) p(\mu)  p(z) \\
             & = \prod_{i=1}^N p(x_i | \mu_{z_i}, z_i) p(\mu_{z_i}) p(z_i) \\
\end{aligned} \tag{2}
$$

Now, if we encode $z_i$ as a one-hot vector $z_i = [0, 0, ..., 1, ..., 0]$, where the $1$ is in the $k$th position, and the generative process can be represented by the following full hierarchical model:

$$
\begin{aligned}
\mu_k & \sim \mathcal{N}(0, \sigma^2), \quad &  k = 1, ..., K \\
z_i & \sim \text{Categorical}(1/K, \cdots, 1/K), \quad & i = 1, ..., N \\
x_i | z_i, \mu & \sim \mathcal{N}(z_i^T \mu, 1), \quad & i = 1, ..., N
\end{aligned} \tag{3}
$$

Then the joint distribution of the model can be written as:

$$
\begin{aligned}
p(x, \mu, z) & = p(x| \mu, z) p(\mu) p(z) \\
             & = \prod_{i=1}^N p(x_i | \mu, z_i) p(\mu) p(z_i) \\
\end{aligned}
$$

_Remark_: The one-hot vector $z_i$ is a vector of length $K$ with all zeros except for a single one in the $k$th position. The $k$th position indicates the cluster assignment of the $i$th data point $x_i$. The mean vector $\mu$ is a vector of length $K$ with each element $\mu_k$ being the mean of the $k$th cluster.

The latent variables are $\{\mu, z\}$, which are $K$ means and $N$ cluster assignments. The observed variables are $\{x\}$, which are $N$ data points.

The evidence (or marginal likelihood) is:

$$
\begin{aligned}
p(x) & =  \int_{\mu} p(\mu) \sum_{z_i} p(x_i | \mu, z_i) p(z_i) d\mu \\
     & = \sum_{z_i}^K p (z_i) \int_{\mu} p(\mu) p(x_i | \mu, z_i)  d\mu \\
\end{aligned} \tag{4}
$$

Although it is possible to calculate the marginal likelihood $p(x)$, it is computationally expensive: 

- calculate the Gaussian integral $\int_{\mu} p(\mu) p(x_i | \mu, z_i)  d\mu$ for each data point $x_i$ by calculating a Gaussian prior $p(\mu)$ and a Gaussian likelihood $p(x_i | \mu, z_i)$
- sum over all possible cluster assignments $z_i$.

This means that we need to calculate $K^N$ Gaussian integrals (or the complexity is $\mathcal{O}(K^N)$) , which is prohibitively expensive for large $K$ and $N$.

This example illustrates what we mean by the marginal likelihood being intractable.

To solve this problem, we will use variational inference to approximate the marginal likelihood or evidence $p(x)$ by searching for a distribution $q(\mu, z)$ that is close to the true posterior $p(\mu, z | x)$.


## Evidence lower bound (ELBO)

For a general Beyesian model, suppose we have latent variable $z$ and observed variable $x$, the joint distribution of the model is:

$$
p(z | x) = \frac{p(x, z)}{p(x)} = \frac{p(x, z)}{\int_z p(x, z) dz} \tag{5}
$$

Equation (5) gives:

$$
p(x) = \frac{p(x, z)}{p(z | x)} \tag{6}
$$

Now, for the evidence $p(x)$, we can write it as:

$$
\begin{aligned} 
\ln p(x) & = \int_z p(z) dz \ln p(x) \quad \text{$\int_z p(z) dz = 1$} \\
         & = \int_z p(z) dz \ln \frac{p(x, z)}{p(z | x)} \\
         & = \int_z p(z)\ln  \frac{p(x, z)}{p(z | x)}  dz \\
         & = \int_z p(z) \ln \left ( \frac{p(x, z)}{q(z)} \frac{q(z)}{p(z | x)} \right ) dz \\
         & = \int_z p(z) \ln \left ( \frac{p(x, z)}{q(z)} \right ) dz + \int_z p(z) \ln \left ( \frac{q(z)}{p(z | x)} \right ) dz \\
         & = L(x) + KL(q || p) 
\end{aligned} \tag{6}
$$

where $L(x)$ is the evidence lower bound (ELBO) and $KL(q || p)$ is the Kullback-Leibler divergence between the variational distribution $q(z)$ and the true posterior $p(z | x)$, which is non-negative and bounded by zero.




{% endkatexmm %}