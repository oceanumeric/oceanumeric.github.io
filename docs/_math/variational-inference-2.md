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
\ln p(x) & = \int_z q(z) dz \ln p(x) \quad \text{$\int_z q(z) dz = 1$} \\
         & = \int_z q(z) dz \ln \frac{p(x, z)}{p(z | x)} \\
         & = \int_z q(z)\ln  \frac{p(x, z)}{p(z | x)}  dz \\
         & = \int_z q(z) \ln \left ( \frac{p(x, z)}{q(z)} \frac{q(z)}{p(z | x)} \right ) dz \\
         & = \int_z q(z) \ln \left ( \frac{p(x, z)}{q(z)} \right ) dz + \int_z q(z) \ln \left ( \frac{q(z)}{p(z | x)} \right ) dz \\
         & = L(x) + KL(q || p) 
\end{aligned} \tag{6}
$$

where $L(x)$ is the evidence lower bound (ELBO) and $KL(q || p)$ is the Kullback-Leibler divergence between the variational distribution $q(z)$ and the true posterior $p(z | x)$, which is non-negative and bounded by zero.

As before, we can maximize the ELBO $L(x)$ to maximize the evidence $p(x)$ by optimization with respect to the variational distribution $q(z)$, which is equivalent to minimizing the KL divergence $KL(q || p)$. Since working with the true posterior $p(z | x)$ is intractable, we therefore consider a family of variational distributions $q(z)$ that are easy to work with, which will make the optimization of the ELBO $L(x)$ tractable.



Now, for the ELBO $L(x)$, we have:

$$
\begin{aligned}
L(x) & = \int_z q(z) \ln \left ( \frac{p(x, z)}{q(z)} \right ) dz \\
    & = \int_z q(z) \ln \left ( \frac{p(x|z)p(z)}{q(z)} \right ) dz \\
    & = \int_z q(z) \left [ \ln p(x|z) + \ln \frac{p(z)}{q(z)}   \right] dz  \\
    & = \int_z q(z) \ln p(x|z) dz - \int_z q(z) \ln \frac{q(z)}{p(z)} dz \\
    & = \mathbb{E}_{q(z)} \left [ \ln p(x|z) \right ] - KL(q(z) || p(z)) \\
    & = \mathbb{E}_{q(z)} \left [ \ln \frac{p(x, z)}{p(z)} \right ] - KL(q(z) || p(z)) \\
    & = \mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ] - \mathbb{E}_{q(z)} \left [ \ln p(z) \right ] - KL(q(z) || p(z)) \\
    & = \mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ] - \mathcal{H}(q(z)) - KL(q(z) || p(z)) 
\end{aligned} \tag{7}
$$


The ELBO $L(x)$ is the sum of three terms:

- the first term $\mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ]$ is the expected log joint probability of the observed and latent variables under the variational distribution $q(z)$
- the second term $\mathcal{H}(q(z))$ is the entropy of the variational distribution $q(z)$
- the third term $KL(q(z) || p(z))$ is the Kullback-Leibler divergence between the variational distribution $q(z)$ and the prior $p(z)$.

The intuition behind the ELBO $L(x)$ is that we can condition on the latent variable $z$ and maximize the expected log joint probability $\mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ]$ to maximize the ELBO $L(x)$, which is equivalent to maximizing the evidence $p(x)$. This means _we do not have to consider the intractable marginal likelihood $p(x)$ directly_. 

How could we choose the variational distribution $q(z)$? We can choose the variational distribution $q(z)$ to be a member of the exponential family, which is a family of distributions that are easy to work with. The coming section will show how to choose the variational distribution $q(z)$.


## Mean field theory 

Mean field theory is a variational inference method that assumes that the latent variables $z$ are independent of each other. This means that we can factorize the variational distribution $q(z)$ as:

$$
q(z) = \prod_{i=1}^N q_i(z_i) \tag{8}
$$

where $q_i(z_i)$ is the variational distribution for the $i$-th latent variable $z_i$.

## Coordinate ascent mean-field variational inference

By factorizing the variational distribution $q(z)$, we can maximize the ELBO $L(x)$ by coordinate ascent. This means that we can maximize the ELBO $L(x)$ with respect to each variational distribution $q_i(z_i)$ while holding the other variational distributions $q_j(z_j)$ for $j \neq i$ fixed. This is equivalent to minimizing the KL divergence $KL(q || p)$ with respect to each variational distribution $q_i(z_i)$ while holding the other variational distributions $q_j(z_j)$ for $j \neq i$ fixed.

Here is the algorithm for coordinate ascent mean-field variational inference:

1. choose a family of variational distributions $q(z)$
2. compute the ELBO $L(x)$
3. maximize the ELBO $L(x)$ with respect to each variational distribution $q_i(z_i)$ while holding the other variational distributions $q_j(z_j)$ for $j \neq i$ fixed
4. repeat step 2 and 3 until convergence

The reason we could do this is that the factorization of the variational distribution $q(z)$ allows us to optimize each variational distribution $q_i(z_i)$ independently and the computation becomes tractable with log-likelihoods and expectations (the production in equation (8) becomes a summation).


## Applying VI to the Bayesian mixture of Gaussians


Now, let's apply variational inference to the Bayesian mixture of Gaussians. First, let's review our model:

$$
\begin{aligned}
\mu_k & \sim \mathcal{N}(0, \sigma^2), \quad &  k = 1, ..., K \\
z_i & \sim \text{Categorical}(1/K, \cdots, 1/K), \quad & i = 1, ..., N \\
x_i | z_i, \mu & \sim \mathcal{N}(z_i^T \mu, 1), \quad & i = 1, ..., N
\end{aligned}  \tag{9}
$$

In equation (7), we have shown 

$$
\begin{aligned}
L(x) & = \mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ]  - \mathbb{E}_{q(z)} \left [ \ln p(z) \right ] - KL(q(z) || p(z)) \\
    & = \mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ] - \mathcal{H}(q(z)) - KL(q(z) || p(z)) \\
    & = \arg \max_{q(z)} \mathbb{E}_{q(z)} \left [ \ln p(x, z) \right ] - \mathcal{H}(q(z)) 
\end{aligned}  \tag{10}
$$

__Note__: equation (10) gives the general form of the ELBO $L(x)$ for any model, which means we only have one latent variable $z$ in equation (10). In our Bayesian mixture of Gaussians, we have two latent variables. 


We have two latent variables: the cluster assignment $z_i$ for each data point $x_i$ and the cluster mean $\mu_k$ for each cluster $k$. We have $N$ data points and $K$ clusters. The cluster assignment $z_i$ is a one-hot vector, which means that $z_{ik} = 1$ if data point $x_i$ is assigned to cluster $k$ and $z_{ik} = 0$ otherwise. The cluster mean $\mu_k$ is a $D$-dimensional vector, where $D$ is the dimension of the data points $x_i$.

For latent variables $\{z, \mu\}$, we will choose the following variational distribution:

$$
q(\mu, z) = q(\mu; m, s^2) q(z; \phi) = \prod_j q(\mu_j; m_j, s_j^2) \prod_i q(z_i; \phi_i) \tag{11}
$$

where 

$$
\begin{aligned}
q(\mu_j; m_j, s_j^2) & = \mathcal{N}(m_j, s_j^2) \\
q(z_i; \phi_i) & = \text{Categorical}(\phi_i)
\end{aligned} \tag{12}
$$

### The first term in the ELBO $L(x)$

First, let's write down the joint probability of the observed and latent variables, which is the first term in the ELBO $L(x)$ (equation (10)):

$$
\begin{aligned}
\ln p(x, z, \mu) & = \ln p(\mu) + \ln p(z) + \ln p(x|z, \mu) \\
                 & = \sum_j \ln p(\mu_j) + \sum_i [ \ln p(z_i) + \ln p(x_i|z_i, \mu) ]\\
                 & = \sum_j \ln p(\mu_j) + \sum_i \ln p(x_i|z_i, \mu) + \text{const}
\end{aligned} \tag{13}
$$

_Remark_: the last line is because $z_i \sim \text{Categorical}(1/K, \cdots, 1/K)$, which means that $\ln p(z_i)$ is a constant.

Now, we will compute $\ln p(\mu_j)$:

$$
\begin{aligned}
\ln p(\mu_j) & = \ln  \left [ \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left(- \frac{\mu_j^2}{2\sigma^2} \right )  \right ] \quad \text{based on equation (9)}  \\
             & = - \frac{\mu_j^2}{2\sigma^2} + \text{const} \\
             & \propto - \frac{\mu_j^2}{2\sigma^2}
\end{aligned} \tag{14}
$$

For $\ln p(x_i|z_i, \mu)$, we have:

$$
\begin{aligned}
\ln p(x_i |z_i, \mu) & = \ln p(x_i| z_i) p(x_i | \mu) \\
                     & = \ln \prod_k p(x_i | \mu_k)^{z_{ik}} \quad \text{using the one-hot vector property of } z_i \\
                     & = \sum_k z_{ik} \ln p(x_i | \mu_k) \\
                     & = \sum_k z_{ik} \ln \left [ \frac{1}{\sqrt{2\pi}} \exp \left(- \frac{(x_i - \mu_k)^2}{2} \right )  \right ] \\
                    & = \sum_k z_{ik} \left [ - \frac{(x_i - \mu_k)^2}{2} + \text{const} \right ] \\
                    & \propto \sum_k - z_{ik} \frac{ (x_i - \mu_k)^2}{2}
\end{aligned} \tag{15}
$$

Now, we will compute full joint distribution by substituting equations (14) and (15) into equation (13):

$$
\ln p(x, z, \mu) \propto \sum_j \left [ - \frac{\mu_j^2}{2\sigma^2} \right ] - \sum_i \sum_k z_{ik} \frac{(x_i - \mu_k)^2}{2}\tag{15}
$$

### Entropy of the variational distribution

Now, we will compute the entropy of the variational distribution $q(\mu, z)$, which is the second term in the ELBO $L(x)$ (equation (10)):

$$
\begin{aligned}
\ln q(\mu, z) & = \ln q(\mu) + \ln q(z) \\
              & = \sum_i \ln q(z_i) + \sum_j \ln q(\mu_j) \\
              & = \sum_i \ln \text{Categorical}(\phi_i) + \sum_j \ln \mathcal{N}(m_j, s_j^2) \quad \text{based on equation (11)} \\
\end{aligned} \tag{16}
$$





{% endkatexmm %}