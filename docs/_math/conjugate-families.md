---
title: Conjugate Families
subtitle: When we build a model, we need to choose a prior distribution. If we choose a prior distribution from the same family as the posterior distribution, we can use the posterior distribution as the new prior distribution. This is called a conjugate prior. In this post, we will look at some of the most common conjugate priors.
layout: math_page_template
date: 2023-04-11
keywords: probabilistic-thinking machine-learning bayesian-inference bayesian-statistics conjugate-prior conjugate-family
published: true
tags: probability algorithm data-science machine-learning binomial-distribution bayesian-statistics beta-distribution conjugate-prior normal-distribution
---

Having some conjugate priors in our toolbox is very useful. In this post, we will look at some of the most common conjugate priors.


## Gamma-Poisson conjugate family


{% katexmm %}


A Poisson distribution is a _discrete_ distribution which can get any non-negative integer values. It is a natural distribution for modelling counts, such as goals in a football game, or a number of bicycles passing a certain point of the road in one day, or a number of people who visit a website in one day, or a number of fraud call you receive in one day. The Poisson distribution is defined by a single parameter $\lambda$, which is the expected value of the distribution. The _probability mass function_ of the Poisson distribution is given by

$$
X | \lambda \sim \text{Poisson}(\lambda) \quad \text{with} \quad f(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \ \ \text{for} \ k = 0, 1, 2, \ldots
$$

The Poisson distribution has equal expected value and variance, which is $\lambda$:

$$
\mathbb{E}[X] = \mathbb{V}[X] = \lambda
$$



{% endkatexmm %}