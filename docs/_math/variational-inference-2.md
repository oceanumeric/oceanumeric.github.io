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

- [Big picture again](#big-picture-again)
- [Bayesian mixture of Gaussians](#bayesian-mixture-of-gaussians)
- [Evidence lower bound (ELBO)](#evidence-lower-bound-elbo)
- [Mean field theory](#mean-field-theory)
- [Coordinate ascent mean-field variational inference](#coordinate-ascent-mean-field-variational-inference)
- [Applying VI to the Bayesian mixture of Gaussians](#applying-vi-to-the-bayesian-mixture-of-gaussians)
- [Implementation](#implementation)
- [Summary and reflection](#summary-and-reflection)


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
\ln p(x, z, \mu) \propto \sum_j \left [ - \frac{\mu_j^2}{2\sigma^2} \right ] - \sum_i \sum_k z_{ik} \frac{(x_i - \mu_k)^2}{2}\tag{16}
$$

### Entropy of the variational distribution

Now, we will compute the entropy of the variational distribution $q(\mu, z)$, which is the second term in the ELBO $L(x)$ (equation (10)):

$$
\begin{aligned}
\ln q(\mu, z) & = \ln q(\mu) + \ln q(z) \\
              & = \sum_i \ln q(z_i) + \sum_j \ln q(\mu_j) \\
              & = \sum_i \ln \text{Categorical}(\phi_i) + \sum_j \ln \mathcal{N}(m_j, s_j^2) \quad \text{based on equation (11)} \\
              & = \sum_i \ln \left [ \prod_k \phi_{ik}^{z_{ik}} \right ] + \sum_j \ln \left [ \frac{1}{\sqrt{2\pi s_j^2}} \exp \left ( - \frac{(\mu_j - m_j)^2}{2s_j^2} \right ) \right ] \\
              & = \sum_i \sum_k z_{ik} \ln \phi_{ik} + \sum_j \left [ -\frac{1}{2} \ln (2\pi s_j^2)  - \frac{(\mu_j - m_j)^2}{2s_j^2}  \right ] \\
\end{aligned} \tag{17}
$$

Therefore, the ElBO $L(x)$ is:

$$
\begin{aligned}
L(x) \propto & \sum_j \mathbb{E_q} \left  [ - \frac{\mu_j^2}{2\sigma^2} \right ] - \sum_i \sum_k \mathbb{E_q}  \left [ z_{ik} \frac{(x_i - \mu_k)^2}{2} \right ] + \\
     & \sum_i \sum_k z_{ik} \mathbb{E_q} [ \ln \phi_{ik}] + \sum_j \mathbb{E_q}  \left [ -\frac{1}{2} \ln (2\pi s_j^2)  - \frac{(\mu_j - m_j)^2}{2s_j^2}  \right ]
\end{aligned} \tag{18}
$$

With equation (18) we can maximize the ELBO $L(x)$ with respect to the variational parameters $\phi$ and $m, s^2$ by taking the derivatives of $L(x)$ with respect to $\phi$ and $m, s^2$ and set them to zero.

We will not go through the derivation of the update equations for $\phi$ and $m, s^2$ here. Instead, we will just write down the update equations:


$$
\begin{aligned}
\phi_{ik}^* & \propto \exp \left [ - \frac{1}{2} (m_j^2 + s_j^2) + x_im_j \right ] \\
m_j^* & = \frac{\sum_i \phi_{ij}x_i}{\frac{1}{\sigma^2} + \sum_{i} \phi_{ij}} \\
s_j^2 & = \frac{1}{\frac{1}{\sigma^2} + \sum_{i} \phi_{ij}}
\end{aligned} \tag{19}
$$

> **Note**: you probably notice that the derivation is very long and tedious. In practice, it is difficult to scale up the derivation to more complicated models. [Some people](https://www.inference.vc/online-bayesian-deep-learning-in-production-at-tencent/){:target="_blank"} argued that Bayesian deep learning is not a practical tool but a theoretical curiosity. However, there are some recent works that try to make Bayesian deep learning more practical. For example, [this paper](https://arxiv.org/abs/1806.05978){:target="_blank"} proposed a new variational inference method that can scale up to large models. For probabilistic programming, [this paper](https://arxiv.org/pdf/1301.1299){:target="_blank"} and more recent [this paper](https://arxiv.org/abs/1603.00788){:target="_blank"} proposed a new method that can automatically derive the update equations for the variational parameters. Numpyro is a probabilistic programming library that could do automatic differentiation and automatic derivation of the update equations for the variational parameters. 


## Implementation

Now, we will implement the above model in Python. The following figure gives the histogram of simulated data.

<div class='figure'>
    <img src="/math/images/variational-inference-2.png"
         alt="Entropy illustration"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The histogram and density plot of three Gaussian distributions.
    </div>
</div>


```python
class UGMM:
    """
    Univariate Gaussian Mixture Model
    """
    
    def __init__(self, X, K = 2, sigma = 1):
        self.X = X
        self.K = K
        self.N = X.shape[0]
        self.sigma2 = sigma**2
        
        # initialize the parameters
        # using dirichlet distribution to initialize the prior probability
        # we fix alpha in the range of [1, 10] for initialization
        # it can be changed to other values
        alpha_const = np.random.random()*np.random.randint(1, 10)
        self.phi = np.random.dirichlet([alpha_const]*self.K, size=self.N)
        # initialize the mean from uniform distribution
        self.m = np.random.uniform(min(self.X), max(self.X), self.K)
        # initialize the variance from uniform distribution
        self.s2 = np.random.uniform(0, 1, self.K)
        
    def _get_elbo(self):
        # calculate the evidence lower bound
        # term 1 in euqation (14)
        # although we use sigma^2 in equation (14) but we use s2 in the code
        # because we are not estimating sigma^2 but s2 (variational inference)
        elbo_term1 = np.log(self.s2) - self.m / self.s2
        elbo_term1 = elbo_term1.sum()
        # term is not exactly the same as equation (14)
        # herer we penalize the model with large variance
        # term 2 based on equation (17)
        # again the term is not exactly the same as equation (17)
        # but proportional to it
        elbo_term2 = -0.5 * np.add.outer(self.X**2, self.s2+self.m**2)
        elbo_term2 += np.outer(self.X, self.m)
        elbo_term2 -= np.log(self.phi)
        elbo_term2 *= self.phi
        elbo_term2 = elbo_term2.sum()
        
        return elbo_term1 + elbo_term2
    
    def _update_phi(self):
        t1 = np.outer(self.X, self.m)
        t2 = -(0.5*self.m**2 + 0.5*self.s2)
        exponent = t1 + t2[np.newaxis, :]
        self.phi = np.exp(exponent)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]
        
    def _update_m(self):
        self.m = (self.phi*self.X[:, np.newaxis]).sum(0) * (1/self.sigma2 + self.phi.sum(0))**(-1)
        assert self.m.size == self.K
        #print(self.m)
        self.s2 = (1/self.sigma2 + self.phi.sum(0))**(-1)
        assert self.s2.size == self.K
        
    def _cavi(self):
        self._update_phi()
        self._update_m()
    
    def fit(self, max_iter=100, tol=1e-10):
        # fit the model
        self.elbos = [self._get_elbo()]
        self.track_m = [self.m.copy()]
        self.track_s2 = [self.s2.copy()]
        
        for iter_ in range(1, max_iter+1):
            self._cavi()
            self.track_m.append(self.m.copy())
            self.track_s2.append(self.s2.copy())
            self.elbos.append(self._get_elbo())
            
            if iter_ % 10 == 0:
                print("Iteration: {}, ELBO: {}".format(iter_, self.elbos[-1]))
                
            if np.abs(self.elbos[-1] - self.elbos[-2]) < tol:
                # print convergence information at iteration i
                print("Converged at iteration: {}, ELBO: {}".format(iter_,
                                                                        self.elbos[-1]))
                break
    
    
def test_univariate_gmm():
    # test ugmm with 3 clusters
    np.random.seed(42)
    num_components = 3
    mu_arr = np.random.choice(np.arange(-10, 10, 2),
                        num_components) + np.random.random(num_components)
    sample_size = 1000
    X = np.random.normal(loc=mu_arr[0], scale=1, size=sample_size)
    for i, mu in enumerate(mu_arr[1:]):
        X = np.append(X, np.random.normal(loc=mu, scale=1, size=sample_size))
        
    # plot the data
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.histplot(X[:sample_size], ax=ax, kde=True)
    sns.histplot(X[sample_size:sample_size*2], ax=ax, kde=True)
    sns.histplot(X[sample_size*2:], ax=ax, kde=True)
    
    # initialize the model
    ugmm = UGMM(X, K=3)
    ugmm.fit()
    
    # print out the true mean and estimated mean
    print("True mean: \n", sorted(mu_arr))
    print("Estimated mean: \n", sorted(ugmm.m))

# Iteration: 10, ELBO: -3574.243674099098
# Iteration: 20, ELBO: -3574.21530393399
# Iteration: 30, ELBO: -3574.21530234368
# Converged at iteration: 32, ELBO: -3574.215302343352
# True mean: 
#  [-3.8439813595575636, 2.5986584841970366, 4.155994520336202]
# Estimated mean: 
#  [-3.775630707652301, 2.634230928126823, 4.142390002370196]
```


## Summary and reflection

In this post, we derived the CAVI algorithm for Gaussian mixture model. We also implemented the algorithm in Python. However, one can see that it is not easy to derive the closed-form update for the parameters. It is also not very intuitive to implement the algorithm even in `Python`. That's why people invented probabilistic programming languages such as `Stan`, `PyMC3`, `Edward`, `Pyro`, `numpyro`, etc. 

Please read [this post](https://www.uber.com/en-DE/blog/pyro/){:target="_blank"} for more information about probabilistic programming languages.

Some people are trying to introduce variational inference in undergraduate statistics courses: [this paper](https://arxiv.org/pdf/2301.01251.pdf){:target="_blank"} and [this repository](https://github.com/oceanumeric/variational_inference_course){:target="_blank"}.

Right now, variational inference is still a very active research area. There are many new algorithms and applications, especially for large datasets. I hope this post can help you understand the basic idea of variational inference.

If you want to study more about probabilistic programming languages, I think `numpyro` is a good choice. It is based on `JAX` and `Pyro`. It is very easy to use and has a very active community. Following a right community is very important for learning new concepts in machine learning and deep learning.



{% endkatexmm %}