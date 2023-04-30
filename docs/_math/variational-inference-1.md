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

- [Big picture](#big-picture)
- [Density estimation](#density-estimation)
- [Jensen’s inequality](#jensens-inequality)
- [EM algorithm](#em-algorithm)
- [Evidence Lower Bound (ELBO)](#evidence-lower-bound-elbo)
- [Application of EM algorithm](#application-of-em-algorithm)
- [Implementation in Python](#implementation-in-python)
- [Summary and reflection](#summary-and-reflection)


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


We could rewrite the lower bound in equation (9) as the following:

$$
\begin{aligned}
\ln p(x; \Theta) &= \sum_{z} q(z) \ln \frac{p(x, z; \Theta)}{q(z)} \\
             &= \sum_{z} q(z) \ln \frac{p(x, z; \Theta)}{p(z|x; \Theta)}  \\ 
             &= \sum_{z} q(z) \ln \frac{p(x, z; \Theta)/q(z)}{p(z|x; \Theta)/q(z)}  \\
             & = \sum_{z} q(z) \ln \frac{p(x, z; \Theta)}{q(z)} - \sum_{z} q(z) \ln \frac{p(z|x; \Theta)}{q(z)}  \\
             & = \sum_{z} q(z) \ln \frac{p(x, z; \Theta)}{q(z)}  + \sum_{z} q(z) \ln \frac{q(z)}{p(z|x; \Theta)}  \\
             & = L(x, \Theta) + KL(q(z) || p(z|x; \Theta))
\end{aligned} \tag{15}
$$ 

Where $L(x, \Theta)$ is the lower bound of the log-likelihood and $KL(q(z) || p(z|x; \Theta))$ is the KL divergence between the posterior distribution $q(z)$ and the true posterior distribution $p(z|x; \Theta)$.

$L(x, \Theta)$ is also called the evidence lower bound (ELBO). The ELBO is a lower bound of the log-likelihood. The KL divergence is always non-negative, which means that the ELBO is always smaller than the log-likelihood. 


$$
\ln p(x; \Theta) \geq L(x, \Theta) \tag{16}
$$


Therefore, we could maximize the ELBO to maximize the log-likelihood.


## Application of EM algorithm

Now, we could apply the EM algorithm to Gaussian mixture model (GMM). Suppose we have some data $x_1, x_2, \cdots, x_n$, which is from $K$ Gaussian distributions (K mixture components). To estimate the parameters of the GMM, we could use the EM algorithm. 

Let's set up our notation first:

- $\mu_k$ is the mean of the $k$-th Gaussian distribution.
- $\Sigma_k$ is the covariance matrix of the $k$-th Gaussian distribution.
- $\phi_k$ is the mixing coefficient of the $k$-th Gaussian distribution.
- $z_i$ is the latent variable of the $i$-th data point. $z_i$ is a one-hot vector, which means that $z_{ik} = 1$ if the $i$-th data point is from the $k$-th Gaussian distribution. Otherwise, $z_{ik} = 0$.

_Remark_: $x_i$ does not have to be a scalar. It could be a vector such as $x_i \in \mathbb{R}^d$.

Our goal is to maximize the log-likelihood of the GMM: 

$$
\arg \max_{\mu, \Sigma, \phi} \sum_{i=1}^n \ln p(x_i; \mu, \Sigma, \phi) \tag{17}
$$


__E-step__: In the E-step, we calculate the posterior distribution of the latent variable $z$ given the data $x$ and the parameter $\Theta$:

$$
q(z_i) = p(z_i | x_i; \Theta) = p(z_i | x_i; \mu, \Sigma, \phi) \tag{18}
$$


__M-step__: In the M-step, we maximize the lower bound with respect to the parameter $\Theta$. The reason why we could maximize the lower bound is that the KL divergence is always non-negative. Therefore, we could maximize the lower bound to maximize the log-likelihood, which makes the optimization process tractable.

According to equation (14), we could maximize the lower bound with respect to the parameter $\Theta$:

$$
\begin{aligned}
\Theta^* & = \arg \max_\Theta \sum_{i=1}^n \sum_{j}^K  q(z_i= j) \ln \frac{p(x_i, z_i ; \Theta)}{q_i(z_i = j)} \\
         & = \arg \max_\Theta \sum_{i=1}^n \sum_{j}^K  q(z_i= j)\ln \frac{p(x_i|z_i = j; \mu, \Sigma)p(z_i = j; \phi)}{q_i(z_i = j)} \\
\end{aligned} \tag{19}
$$

With the above format, we could leverage the distribution functions of $x_i$ and $z_i$ to calculate the lower bound because

$$
\begin{aligned}
x_i|z_i = j; \mu, \Sigma & \sim  \mathcal{N}(\mu_j, \Sigma_j) \\
z_i & \sim \text{Categorical}(\phi) 
\end{aligned} \tag{20}
$$

Therefore, the equation (19) could be rewritten as:

$$
L := \sum_{i=1}^n \sum_{j=1}^K w_j^{(i)} \ln \frac{\frac{1}{\sqrt{(2\pi)^n|\Sigma_j|}} \exp \left[ -\frac{1}{2}(x_i - \mu_j)'\Sigma_j^{-1}(x_i - \mu_j) \right] \phi_j}{w_j^{(i)}} \tag{21}
$$


With the above equation, we could take the derivative of $ll$ with respect to $\mu_j$, $\Sigma_j$, and $\phi_j$ and set the derivative to zero to find the optimal parameters.

First, let's take the derivative of $L$ with respect to $\mu_j$:

$$
\begin{aligned}
\frac{\partial L}{\partial \mu_j} & = \sum_{i=1}^n \frac{\partial}{\partial \mu_j} \left (  \ln \frac{\frac{1}{\sqrt{(2\pi)^n|\Sigma_j|}}}{w_j^{i}}  + \left[ -\frac{1}{2}(x_i - \mu_j)'\Sigma_j^{-1}(x_i - \mu_j) \right] \right )  \\ 
            & = \sum_{i=1}^n w_j^{(i)} \frac{\partial}{\partial \mu_j} \left[ -\frac{1}{2}(x_i - \mu_j)'\Sigma_j^{-1}(x_i - \mu_j) \right] \\
            & = \frac{1}{2} \frac{\partial}{\partial \mu_j} \sum_{i=1}^n w_j^{(i)} \left [ (\Sigma_j^{-1} + (\Sigma_j^{-1})'(x_i - \mu_j)) \right] \\ 
            & = \sum_{i=1}^n w_j^{(i)} \Sigma_j^{-1}(x_i - \mu_j) \\
            & = \Sigma_j^{-1} \sum_{i=1}^n w_j^{(i)} (x_i - \mu_j) \\
            & = 0
\end{aligned} \tag{22}
$$

This gives us the following equation:

$$
\mu_j = \frac{\sum_{i=1}^n w_j^{(i)} x_i}{\sum_{i=1}^n w_j^{(i)}} \tag{23}
$$

Now, let's take the derivative of $L$ with respect to $\Sigma_j$ (the reason that we like log-likelihood is that many terms could be dropped):

$$
\begin{aligned}
\frac{\partial L}{\partial \Sigma_j} & = \frac{\partial}{\partial \Sigma_j} \left [ \sum_{i=1}^n w_j^{(i)} \left ( \ln \frac{1}{\sqrt{(2\pi)^n}} + \ln \frac{1}{\sqrt{|\Sigma_j|}} - \ln w_j^{(i)} -\frac{1}{2}(x_i - \mu_j)'\Sigma_j^{-1}(x_i - \mu_j) \right)  \right ] \\
& =  \frac{\partial}{\partial \Sigma_j} \left [ \sum_{i=1}^n w_j^{(i)} \left ( \ln \frac{1}{\sqrt{|\Sigma_j|}}  -\frac{1}{2}(x_i - \mu_j)'\Sigma_j^{-1}(x_i - \mu_j) \right)  \right ] \\
& = \sum_{i=1^n}w_j^{(i)} \left[ \Sigma_j^{-1} - \Sigma_j^{-1}  (x_i - \mu_j)(x_i - \mu_j)' \Sigma_j^{-1} \right] \\ 
& = 0
\end{aligned}
$$

This gives us the following equation:

$$
\Sigma_j = \frac{\sum_{i=1}^n w_j^{(i)} (x_i - \mu_j)(x_i - \mu_j)'}{\sum_{i=1}^n w_j^{(i)}} \tag{24}
$$

Finally, let's take the derivative of $L$ with respect to $\phi_j$. Since there is a constraint that $\sum_{j=1}^K \phi_j = 1$, we could use the Lagrange multiplier to solve this problem:

$$
\begin{aligned}
\mathcal{L}(\phi) = L + \lambda \left ( \sum_{j=1}^K \phi_j - 1 \right ) \tag{25}
\end{aligned}
$$ 

Again for $L$ we could drop many terms when it comes to the derivative of $\phi_j$:

$$
L = \sum_{i=1}^n \sum_{j=1}^K w_j^{(i)} \ln \phi_j \tag{26}
$$

The derivative of $\mathcal{L}$ with respect to $\phi_j$ is:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \phi_j} & = \frac{\partial}{\partial \phi_j} \left [ \sum_{i=1}^n w_j^{(i)} \ln \phi_j + \lambda \left ( \sum_{j=1}^K \phi_j - 1 \right ) \right ] \\
& = \frac{\sum_{i=1}^n w_j^{(i)}}{\phi_j} + \lambda \\
& = 0
\end{aligned} 
$$

This gives the following equation:

$$
\phi_j = -\frac{\sum_{i=1}^n w_j^{(i)}}{\lambda} 
$$

Since $\sum_{j=1}^K \phi_j = 1$, we have:

$$
\begin{aligned}
\sum_{j=1}^K \phi_j & = -\sum_{j=1}^K \frac{\sum_{i=1}^n w_j^{(i)}}{\lambda} = 1 \\
\lambda & = -\sum_{j=1}^K \sum_{i=1}^n w_j^{(i)} = -\sum_{i=1}^n \sum_{j=1}^K w_j^{(i)} = -\sum_{i=1}^n 1 = -n
\end{aligned}
$$

Therefore, we have:

$$
\phi_j = \frac{\sum_{i=1}^n w_j^{(i)}}{n} \tag{27}
$$


## Implementation in Python

Now, let's implement the EM algorithm in Python. If you look at the equations above, you should notice that we have to calculate $w_j^{(i)}$ in each iteration, which is the probability that $x_i$ belongs to the $j$-th Gaussian distribution. This is the **responsibility** of the $j$-th Gaussian distribution for the $i$-th data point. This is the posterior probability of the $j$-th Gaussian distribution given the $i$-th data point. The value of $w_j^{(i)}$ is calculated as follows:

$$
q(z_i) = w_j^{(i)} = \frac{\phi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}{\sum_{k=1}^K \phi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)} \tag{28}
$$

Because we do not know $\phi_j$, this parameter is prior to the EM algorithm. We could initialize it with some random values. In the following code, we initialize $\phi_j$ with $1/K$.

```python
class GMM:
    """
    Gaussian Mixture Model with EM algorithm
    
    It is a semi-supervised learning algorithm, which means user need to provide 
    the number of clusters.
    """
    
    def __init__(self, X, k=2):
        # set x as array
        X = np.array(X)
        self.n, self.m = X.shape  # n: sample size, m: feature size
        self.data = X.copy()
        self.k = k  # number of clusters
        
        # initialize parameters for EM algorithm
        
        # initialize the mean vector as random vector for each cluster
        self.mean = np.random.rand(self.k, self.m)
        # initialize the covariance matrix as identity matrix for each cluster
        self.sigma = np.array([np.eye(self.m)] * self.k)
        # initialize the prior probability as equal for each cluster
        self.phi = np.ones(self.k) / self.k
        # initialize the posterior probability as zero
        self.w = np.zeros((self.n, self.k))
        
    def _gaussian(self, x, mean, sigma):
        
        pdf = sp.stats.multivariate_normal.pdf(x, mean=mean, cov=sigma)
        
        return pdf
        
    
    def _e_step(self):
        # calculate the posterior probability based on equation (28)
        for i in range(self.n):
            density = 0 # initialize the density
            for j in range(self.k):
                temp = self.phi[j] * self._gaussian(self.data[i],
                                                        self.mean[j],
                                                        self.sigma[j])
                # update the density (marginal probability)
                density += temp
                # update the posterior probability (joint probability)
                self.w[i, j] = temp
            # normalize the posterior probability
            self.w[i] /= density
            # assert the sum of posterior probability is 1
            assert np.isclose(np.sum(self.w[i]), 1)
            
    def _m_step(self):
        # update the parameters
        for j in range(self.k):
            # get the sum of posterior probability for each cluster
            sum_w = np.sum(self.w[:, j])
            # update the prior probability based on equation (27)
            self.phi[j] = sum_w / self.n
            # update the mean vector based on equation (23)
            self.mean[j] = np.sum(self.w[:, j].reshape(-1, 1) * self.data,
                                                    axis=0) / sum_w
            # update the covariance matrix based on equation (24)
            self.sigma[j] = np.dot(
                    (self.w[:, j].reshape(-1, 1) * (self.data - self.mean[j])).T,
                                (self.data - self.mean[j])) / sum_w
            
    def _fit(self):
        self._e_step()
        self._m_step()
        
    def loglikelihood(self):
        # calculate the loglikelihood based on equation (21)
        ll = 0
        for i in range(self.n):
            temp = 0
            for j in range(self.k):
                temp += self.phi[j] * self._gaussian(self.data[i],
                                                        self.mean[j],
                                                        self.sigma[j])
            ll += np.log(temp)
            
        return ll
    
    def fit(self, max_iter=100, tol=1e-6):
        # initialize the loglikelihood
        ll = [self.loglikelihood()]
        # initialize the number of iteration
        i = 0
        # initialize the difference between two loglikelihood
        diff = 1
        # iterate until the difference is less than tolerance or reach the max iteration
        while diff > tol and i < max_iter:
            # update the parameters
            self._fit()
            # calculate the loglikelihood
            ll.append(self.loglikelihood())
            # calculate the difference
            diff = np.abs(ll[-1] - ll[-2])
            # update the number of iteration
            i += 1
            # print the loglikelihood every 2 iterations
            if i % 2 == 0:
                print("Iteration: {}, loglikelihood: {}".format(i, ll[-1]))
    

def test_gmm():
    """
    Test GMM class
    """
    # set seed
    np.random.seed(57)
    # generate a mixture of two normal distributions
    # with sample size 30 and 70 respectively
    # one normal distribution has mean (0, 3) and the other has mean (10, 5)
    # one normal distribution has covariance matrix [[0.5, 0], [0, 0.8]]
    # the other normal distribution has identity covariance matrix
    
    X = np.concatenate(
                (np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 30),
                    np.random.multivariate_normal([10, 5], np.eye(2), 70))
                )
    print("If we treat the data as one cluster:")
    print(X.shape, X.mean(axis=0), X.std(axis=0))
    
    print("-" * 60)
    print("Now, we use GMM to fit the data with 2 clusters:")
    
    gmm = GMM(X, k=2)
    gmm.fit()
    
    # print out the parameters
    print("Mean: \n", gmm.mean)
    print("Covariance matrix: \n", gmm.sigma)
    print("Prior probability: \n", gmm.phi)
    # print("Posterior probability: \n", gmm.w)


# If we treat the data as one cluster:
# (100, 2) [6.82376692 4.47781942] [4.55483689 1.36784028]
# ------------------------------------------------------------
# Now, we use GMM to fit the data with 2 clusters:
# Iteration: 2, loglikelihood: -436.6134973400012
# Iteration: 4, loglikelihood: -422.43789313300357
# Iteration: 6, loglikelihood: -341.4412273815827
# Iteration: 8, loglikelihood: -337.46812095035875
# Mean: 
#  [[9.74569874e+00 5.05825309e+00]
#  [5.92600895e-03 3.12347417e+00]]
# Covariance matrix: 
#  [[[0.94691865 0.09556468]
#   [0.09556468 1.08137946]]

#  [[0.54143237 0.04580301]
#   [0.04580301 1.09304612]]]
# Prior probability: 
#  [0.7 0.3]
```

By implementing the GMM algorithm, we can see that the mean vectors and covariance matrices are close to the true values. The prior probability is also close to the true values. However, the covariance matrix of the first cluster is not close to the true value. This is because the sample size of the first cluster is small. If we increase the sample size of the first cluster, the covariance matrix of the first cluster will be close to the true value.


## Summary and reflection

I hope after reading this article, you can understand the EM algorithm better. The EM 
algorithm is a Bayesian algorithm. It is a Bayesian algorithm because it uses the
posterior probability to update the parameters. 

Let's take a look at our model again:

$$
p(x, z; \Theta),
$$

where $\Theta = (\mu, \Sigma, \phi)$ is the parameter set, $z$ is the random variable
that represents the cluster, and $x$ is the random variable that represents the data. If you print out the posterior probability $w$ in our GMM example, you will find that the posterior probability is close to 1 or 0, which is just a _one-hot vector that maps the data to the cluster_. At the same time, $\phi$ is the prior probability of each cluster.


At the beginning, we initialize $\phi$ as a uniform distribution, which means that we
assume that each cluster has the same probability. Then, we use the posterior probability
to update $\phi$. 

Although we use a simple example to explain the EM algorithm, the Bayesian idea behind
this algorithm is very important. If you understand this example well, then you will
have a better understanding of the latent variable model.

__Note__: The code in this article could only work for the simple example with two
clusters. If you want to use it for other examples, you need to modify the code or use the `sklearn` package, which is more efficient and stable.

If you want to have a more rigid understanding of the EM algorithm, you can read the
chapter 9 of the book _Pattern Recognition and Machine Learning_ by Christopher M. Bishop {% cite bishop2006pattern %}.



## Reference

1. [Expectation Maximization](https://zhiyzuo.github.io/EM/#real-example)




{% endkatexmm %}













