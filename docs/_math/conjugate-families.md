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


- [Gamma-Poisson conjugate family](#gamma-poisson-conjugate-family)
- [Normal-Normal Bayesian model](#normal-normal-bayesian-model)

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

Let's say we have a Poisson distribution with parameter $\lambda = 5$. We can generate some random samples from this distribution:

<div class='figure'>
    <img src="/math/images/poisson_dist.png"
         alt="Inequality bounds compare"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The plot of the Poisson distribution with different values of $\lambda$.
    </div>
</div>

In contrast, when events occur at a higher rate of $\lambda = 5$, the model is roughly symmetric and more variable – we’re most likely to observe 4 or 5 events, though have a reasonable chance of observing anywhere between 1 and 10 events.

After learning about the Poisson distribution, we can now look at the exponential distribution. The exponential distribution is a _continuous_ distribution which can get any non-negative real values. It is a natural distribution for modelling time between events, such as the time between two consecutive fraud calls, or the time between two consecutive goals in a football game, or the time between two consecutive visits of a website. The exponential distribution is defined by a single parameter $\lambda$, which is the rate parameter. The _probability density function_ of the exponential distribution is given by


$$
X \sim \text{Exponential}(\lambda) \quad \text{with} \quad f(x) = \lambda e^{-\lambda x} \ \ \text{for} \ x \geq 0, \ \lambda > 0
$$

The exponential distribution has expected value and vaiance given by

$$
\mathbb{E}[X] = \frac{1}{\lambda} \quad \text{and} \quad \mathbb{V}[X] = \frac{1}{\lambda^2}
$$

There are some connections between the Poisson and exponential distributions. The following table shows the relationship between the two distributions:

| Poisson distribution | Exponential distribution |
|-----------------------|--------------------------|
|number of events in a fixed time interval | time between events |
| number of phone calls in a day | time between phone calls |
| discrete | continuous |


<div class='figure'>
    <img src="/math/images/exponential_dist.png"
            alt="Inequality bounds compare"
            style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The plot of the exponential distribution with different values of $\lambda$.
    </div>
</div>
 
The exponential distribution is a special case of the Gamma distribution. The Gamma distribution is a _continuous_ distribution which can get any non-negative real values. It is a natural distribution for modelling time between events, such as the time between two consecutive fraud calls, or the time between two consecutive goals in a football game, or the time between two consecutive visits of a website. The Gamma distribution is defined by two parameters $\alpha$ and $\beta$, which are the shape and rate parameters, respectively. The _probability density function_ of the Gamma distribution is given by

$$
X \sim \text{Gamma}(\alpha, \beta) \quad \text{with} \quad f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x} \ \ \text{for} \ x \geq 0, \ \alpha > 0, \ \beta > 0
$$

where $\alpha$ is even parameter $\alpha = k$, $\beta$ is the rate parameter and the inverse of the scale parameter $\theta = 1 / \beta$. The Gamma distribution has expected value and vaiance given by

$$
\mathbb{E}[X] = \frac{\alpha}{\beta} = k \theta  \quad \text{and} \quad \mathbb{V}[X] = \frac{\alpha}{\beta^2} = k \theta^2
$$

The Gamma distribution is a generalization of the exponential distribution. The exponential distribution is a special case of the Gamma distribution with $\alpha = 1$ and $\beta = \lambda$. This means that the exponential distribution gives the probability of observing the first event in a time interval, while the Gamma distribution gives the probability of observing $k$-th event in a time interval.

<div class='figure'>
    <img src="/math/images/gamma_dist.png"
            alt="Inequality bounds compare"
            style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The plot of the Gamma distribution with different values of $\alpha$ and $\beta$. Note that the x-axis is scaled by $\alpha/\beta$.
    </div>
</div>
 
Now, we will introduce the Gamma-Poisson Bayesian model. 

Let $\lambda > 0$ be an unknown parameter. We assume that $\lambda$ follows a Gamma distribution with parameters $\alpha$ and $\beta$. We also assume that the number of events $X$ follows a Poisson distribution with parameter $\lambda$. The Gamma-Poisson Bayesian model is given by

$$
X_i | \lambda \sim_{i.i.d.} \text{Poisson}(\lambda) \quad \text{and} \quad \lambda \sim \text{Gamma}(\alpha, \beta)
$$

Upon observing $X_1, \ldots, X_n$, we can update our belief about $\lambda$ by using the posterior distribution

$$
\lambda | X_1, \ldots, X_n \sim \text{Gamma}(\alpha + \sum_{i=1}^n X_i, \beta + n)
$$


Now, let's walk through an example of the Gamma-Poisson Bayesian model. Suppose we have a Poisson distribution with parameter $\lambda = 3$. We can generate some random samples from this distribution:

```R
# sample size = 5
n <- 5
lambda_true <- 3
set.seed(123)
y <- rpois(n, lambda_true)
# 2, 4, 2, 5, 6

# chose prior
alpha <- 1
beta <- 1

# set lambda sequence
lambda_seq <- seq(0, 7, 0.1)

options(repr.plot.width = 7, repr.plot.height = 5)
plot(lambda_seq, dgamma(lambda_seq, shape = alpha, rate = beta),
    type = "l", lwd = 2, 
    xlab = "lambda", ylab = "P(lambda)",
    main = "Gamma Prior")
lines(lambda_seq, dgamma(lambda_seq, shape = alpha + sum(y), rate = beta +n),
    type = "l", lwd = 2, col = "blue")
abline(v = lambda_true, lty = 2)
legend('topright', inset = .02, legend = c('prior', 'posterior'),
       col = c('black', 'blue'), lwd = 2)
```

<div class='figure'>
    <img src="/math/images/gamma_prior1.png"
            alt="Inequality bounds compare"
            style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The plot of the Gamma prior and posterior distributions.
    </div>
</div>


We will simulate the Bayesian updating process by drawing different number of observations from the Poisson distribution. As it is shown in figure 4, the posterior distribution concentrate more heavily on the neighborhood of the true parameter value as more observations are drawn.


<div class='figure'>
    <img src="/math/images/gamma_prior2.png"
            alt="Inequality bounds compare"
            style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The plot of the Gamma prior and posterior distributions with different number of observations.
    </div>
</div>


After the first two observations the posterior is still quite close to the prior distribution, but the third observation, which was an outlier, shifts the peak of the posterior from the left side of the mean heavily to the right. But when more observations are drawn, we can observe that the posterior starts to concentrate more heavily on the neighborhood of the true parameter value.


## Normal-Normal Bayesian model

For the Normal-Normal bayesian model, we refer the reader to the [Normal-Normal Bayesian model](https://www.bayesrulesbook.com/chapter-5.html#ch5-n-n-section){:target="_blank"} section of the _Bayes Rules_ book.


## References

1. [Bayes Rules](https://www.bayesrulesbook.com/){:target="_blank"} by 
Alicia A. Johnson, Miles Q. Ott, Mine Dogucu

2. [Conjugate distributions](https://vioshyvo.github.io/Bayesian_inference/conjugate-distributions.html#one-parameter-conjugate-models){:target="_blank"} 

{% endkatexmm %}