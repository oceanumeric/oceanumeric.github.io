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

Much of the content in this post is based on the book by Johnson, et al. {% cite johnson2022bayes %}. Compared to the original chapter in the book, my post will
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

How could we approximate the posterior distribution of $\mu$? The Metropolis-Hastings algorithm relies on the fact that __even if we do not know the posterior model, we do know the
likelihood function and we also know that the posterior distribution is proportional to the
product of the likelihood and the prior__: 

$$  
f(\mu | \bar{y}) \propto f(\mu) L(\mu | y = \bar{y}) \tag{1}
$$

where the likelihood function is:

$$
L(\mu | y = \bar{y}) = \prod_{i=1}^n f(y_i | \mu) = \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left [ - \frac{(y_i - \mu)^2}{2 \sigma^2} \right ]
$$

Like we said before, Monte Carlo Markov Chain (MCMC) methods are based on the idea that we can approximate the posterior distribution by simulating from the posterior distribution. It is all about searching. Then, when it comes to searching, we need two things:

- initial value
- a way to move from one value to another value (direction)

First, let's proposal a uniform distribution for $\mu$ with half-width $w$:

$$
\mu' | \mu \sim \text{Uniform}(\mu - w, \mu + w)
$$

with pdf:

$$
f(\mu' | \mu) = \frac{1}{2w}
$$

With this proposed distribution, we now need to design a mechanism to
decide whether to accept or reject the proposed value $\mu'$, which in the end gives us the direction of the search. The principle of Metropolis-Hastings algorithm is to accept the proposed value $\mu'$ with probability that the chain should spend more time exploring areas of high posterior plausibility _but_ shouldn’t get stuck there forever.


How can we come up a measurement that make sure the chain should spend
more time exploring areas of high posterior plausibility? The answer is
to _use the posterior distribution as a weight_, which is given by the
equation (1).

Here are the steps of the Metropolis-Hastings algorithm:

1. Start with an initial value $\mu_0$.
2. Propose a new value $\mu'$ from the proposal distribution.
3. Compute the acceptance probability
    - if the (unnormalized) posterior probability of $\mu'$ is greater than the (unnormalized) posterior probability of $\mu_0$, then accept $\mu'$ with probability 1, meaning we need to check

    $$
    f(\mu')L(\mu' | y = \bar{y}) > f(\mu_0)L(\mu_0 | y = \bar{y})
    $$
    - otherwise, accept $\mu'$ with probability (maybe go there with a small probability)

With the above steps, we will implement the Metropolis-Hastings algorithm in the coming sections.

## The Metropolis-Hastings algorithm

Before, we present the Metropolis-Hastings algorithm, let's review a formula that we will use in the algorithm:

$$
\mathbb{P}(A \cap B) = \mathbb{P}(A) \mathbb{P}(B | A) = \mathbb{P}(B) \mathbb{P}(A | B) \tag{2}
$$

With this simple formula, we can derive the acceptance probability and implement the Metropolis-Hastings algorithm.

__Metropolis-Hastings algorithm__. Conditioned on data $y$, let parameter $\mu$ has a prior distribution $f(\mu)$ and a likelihood function $L(\mu | y)$. Then, we know that the posterior distribution is proportional to the product of the likelihood and the prior:

$$
f(\mu | y) \propto f(\mu) L(\mu | y) 
$$

A Metropolis-Hastings Markov Chain for $f(\mu | y)$, $\{ \mu_0, \mu_1, \mu_2, \dots \mu_N \}$, evolves as follows. Let $\mu_i$ be the current value of the chain, and let $\mu_{i+1}$ be a proposed value, then we update the chain as follows:

1. Generate a proposed value $\mu'$ from the proposal distribution $q(\mu' | \mu_i)$. (We are not updating the chain yet.)

2. Decide whether to accept $\mu'$ or not by:

    - calculating the acceptance probability $\alpha$:

    $$
    \alpha = \min \left \{ 1,  \frac{f(\mu')L(u'|y)}{f(\mu)L(\mu |y)}\frac{q(\mu | \mu')}{q(\mu' | \mu)}    \right \} \tag{3}
    $$
    - figuratively speaking, flip a weighted coin with probability $\alpha$ to decide whether to accept $\mu'$ or not:

    $$
    \mu_{i+1} =
    \begin{cases}
     \mu' & \text{with probability } \alpha \\
     \mu_i & \text{with probability } 1 - \alpha
    \end{cases} \tag{4}
    $$

For the distribution $q(\mu' | \mu_i)$, we can use a uniform distribution with half-width $w$. Since the proposal distribution is symmetric, we can have:

$$
q(\mu | \mu') = q(\mu' | \mu) = \begin{cases}
    \frac{1}{2w} & \text{when $\mu$ and $\mu'$ are with $w$ units of each other }\\
    0 & \text{otherwise}
\end{cases}
$$

This symmetry means that the chance of proposing a chain move from $\mu$ to $\mu'$ is the same as the chance of proposing a chain move from $\mu'$ to $\mu$. Therefore, we can have:

$$
\alpha = \min \left \{ 1,  \frac{f(\mu')L(u'|y)}{f(\mu)L(\mu |y)}\frac{q(\mu | \mu')}{q(\mu' | \mu)}    \right \} = \min \left \{ 1,  \frac{f(\mu')L(u'|y)}{f(\mu)L(\mu |y)}   \right \} 
$$

Now, apply the formula (2) to the acceptance probability:

$$
\begin{aligned}
\alpha = \min \left \{ 1,  \frac{f(\mu')L(u'|y)}{f(\mu)L(\mu |y)}   \right \}  & = \min \left \{ 1,  \frac{f(\mu')L(u'|y)/ f(y)}{f(\mu)L(\mu |y)/f(y)}   \right \}  \\
& = \min \left \{ 1, \frac{f(\mu'|y)}{f(\mu|y)} \right \} \tag{5}
\end{aligned}
$$

Equation (5) states that although we cannot compute the posterior distribution $f(\mu | y), f(\mu'|y)$,  their _ratio_ is equivalent to that of the unnormalized posterior pdfs (which we can calculate).

Based on equation (5), we can see there are two cases:

- Scenario 1: if $f(\mu'|y) > f(\mu|y)$, then $\alpha = 1$ and we accept $\mu'$.
- Scenario 2: if $f(\mu'|y) < f(\mu|y)$, then $\alpha = f(\mu'|y)/f(\mu|y)$ and we accept $\mu'$ with probability $\alpha$.


Scenario 1 is easy to understand. Scenario 2 is a little bit tricky. We will use an example to illustrate this scenario.











{% endkatexmm %}