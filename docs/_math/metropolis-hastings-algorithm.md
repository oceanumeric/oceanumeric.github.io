---
title: Metropolis-Hastings Algorithm
subtitle: The Metropolis-Hastings algorithm is a Markov chain Monte Carlo (MCMC) algorithm that generates a sequence of random variables from a probability distribution from which direct sampling is difficult.
layout: math_page_template
date: 2023-04-13
keywords: probabilistic-thinking machine-learning bayesian-inference bayesian-statistics conjugate-prior conjugate-family metropolis-hastings-algorithm mcmc
published: true
tags: probability algorithm data-science machine-learning binomial-distribution bayesian-statistics beta-distribution conjugate-prior mcmc 
---

It is beneficial to have a good understanding of the Metropolis-Hastings algorithm, as it is the basis for many other MCMC algorithms. The Metropolis-Hastings algorithm is a Markov 
Chain Monte Carlo (MCMC) algorithm that generates a sequence of random variables from a probability distribution from which direct sampling is difficult.

Much of the content in this section is based on the book by Johnson, et al. {% cite johnson2022bayes %}. Compared to the original chapter in the book, my post will
be more intuitive and less formal.

## The big idea

{% katexmm %}

Considering the following Normal-Normal model with numerical outcome $Y$ that
varies Normally around an _unknown_ mean $\mu$ and _known_ standard deviation 
$\sigma = 0.75$. 

Here is the model:

$$
\begin{aligned}
Y | \mu  &\sim \mathcal{N}(\mu, 0.75^2) \\
\mu &\sim \mathcal{N}(0, 1^2)
\end{aligned}
$$

__Remark:__ The Byesian approach is a hierarchical model, which means:

- The key parameter $\mu$ is a random variable, which is itself drawn from a distribution.
- The random variable $Y$ is drawn from a distribution that depends on $\mu$.

That's why we call $\mu$ a _hyperparameter_.

Let's review the Normal-Normal Bayesian model. Let $\mu$ be an _unknown_ 
mean parameter, and $Y_i$ be a random variable that varies Normally around
$\mu$ with a _known_ standard deviation $\sigma$. The Normal-Normal model
complements the Normal data structure with a Normal prior on $\mu$:

$$
\begin{aligned}
Y_i | \mu  &\sim \mathcal{N}(\mu, \sigma^2) \\
\mu &\sim \mathcal{N}(\theta, \tau^2)
\end{aligned}
$$

Upon observing data $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$, we can compute the posterior distribution of $\mu$:

$$
\mu | \bar{y} \sim \mathcal{N}\left (\theta \frac{\sigma^2}{\sigma^2 + n \tau^2} + \bar{y} \frac{n \tau^2}{\sigma^2 + n \tau^2}, \frac{\sigma^2 \tau^2}{\sigma^2 + n \tau^2} \right )
$$


With the above mode, we can compute the posterior mean and variance of $\mu$:

$$
\mu | (Y = 6.25) \sim \mathcal{N}(4, 0.6^2)
$$

If we cannot calcualte in closed form, we can approximate it using MCMC simuation.

How could approximate the posterior distribution of $\mu$? The Metropolis-Hastings algorithm relies on the fact that __even if we do not know the posterior model, we do know the
likelihood function and we also know that the posterior distribution is proportional to the
product of the likelihood and the prior__: 

$$  
f(\mu | \bar{y}) \propto f(\mu) L(\mu | y = \bar{y})
$$

where the likelihood function is:

$$
L(\mu | y = \bar{y}) = \prod_{i=1}^n f(y_i | \mu) = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left [ - \frac{(y_i - \mu)^2}{2 \sigma^2} \right ]
$$














{% endkatexmm %}