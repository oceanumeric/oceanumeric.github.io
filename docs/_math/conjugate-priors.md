---
title: Conjugate Priors - Binomial Beta Pair
subtitle: Bayesian inference is almost 'everywhere' in data science; with the advance of computational power, it is now possible to apply Bayesian inference to high-dimensional data. In this post, we will discuss the conjugate priors for the binomial distribution.
layout: math_page_template
date: 2023-03-27
keywords: probabilistic-thinking dirichlet-distribution text-mining machine-learning bayesian-inference bayesian-statistics 
published: true
tags: probability algorithm data-science machine-learning binomial-distribution high-dimensional-data bayesian-statistics beta-distribution conjugate-prior
---

Machine learning and deep learning models are trained to learn the parameters from the big data. Very often, knowing how to update the parameters is the key to the success of the model. In this post, we will discuss the conjugate priors for the binomial distribution.

## Binomial Distribution


{% katexmm %}

Almost every one learns about the binomial distribution via the coin flipping experiment. The binomial distribution is a discrete probability distribution that describes the probability of getting $x$ successes in $n$ independent trials with a probability of success $p$. The probability mass function of the binomial distribution is given by

$$
f(x;n,p) = \binom{n}{x}p^x(1-p)^{n-x} \tag{1}
$$

When we use equation (1) in the coin flipping experiment, we know $p = 1/2$. However, there is no way to know $p$ in advance every time when we use this model. 

Let's see a simple example. Image you were a basketball coach and you need to make a decision on whether you should hire a basketball player or not. How could you make the decision? Of course, you will test this player's ability by letting him shoot some free throws. If he makes 10 out of 10 free throws, you will hire him. If he makes 5 out of 10 free throws, you will not hire him.

However, this is not a good way to make the decision. The reason is that the player may be lucky or unlucky. If he is lucky, he will make 10 out of 10 free throws. If he is unlucky, he will make 5 out of 10 free throws. In this case, you will make a wrong decision.

In practice, we could rely on the 'rule of thumb' to make the decision by testing this player several time. However, when it comes to the science, we need to make the decision based on the data and rigorous statistical analysis. 

In this case, we could use the binomial distribution to model the free throw shooting. The probability of success $p$ is the probability of making a free throw. The number of trials $n = 10$ is the number of free throws. The number of successes $x$ is the number of free throws made. Therefore, we could have

$$
f(x;n = 10,p) = \binom{n}{x}p^x(1-p)^{n-x} = \frac{10!}{x!(10-x)!}p^x(1-p)^{10-x} \tag{2}
$$

Unlike flipping a coin, we do not know the probability of success $p$ in advance. However, it is fair to assume that a good player should have a higher probability of success $p$. Therefore, we could use a prior distribution to model the probability of success $p$.

## Prior beliefs

I hope the above example explains why we refer prior as prior beliefs. We can specify our prior beliefs in two ways:

- we can discretize the probability of success $p$ into a finite number of values and assign a probability to each value. For example, we can discretize the probability of success $p$ into 10 values and assign a probability of 0.1 to each value. In this case, we will have a uniform prior distribution.
- we can use a continuous distribution to model the probability of success $p$. For example, we can use a beta distribution to model the probability of success $p$. In this case, we will have a beta prior distribution.

Now, let's assume we have three values of the probability of success $p$ and assign the following probabilities to each value:

$$
\begin{aligned}
\pi(p = 0.7) & = 0.2 \\
\pi(p = 0.5) & = 0.75 \\
\pi(p = 0.1) & = 0.05
\end{aligned}
$$

Note that the probabilities must sum up to 1. The above assignment means that we assume that the basketball player has some probability of being a good player (the probability of success $p = 0.7$ is 0.2), some probability of being an average player (the probability of success $p = 0.5$ is 0.75), and some probability of being a bad player (the probability of success $p = 0.1$ is 0.05). This kind of prior belief is aligned with our intuition as either good plaers or bad players are rare.

Before we move on, let's review Bayes' theorem. Bayes' theorem is a fundamental theorem in probability theory. It is used to update the prior beliefs based on the data. The posterior distribution is the updated prior distribution. The posterior distribution is given by

$$
\begin{aligned}
\pi(p|x) & = \frac{\pi(x|p)\pi(p)}{\pi(x)} \\
& = \frac{\pi(x|p)\pi(p)}{\int_{p} \pi(x|p)\pi(p)dp} \tag{3}
\end{aligned}
$$

where $\pi(p|x)$ is the posterior distribution, $\pi(x|p)$ is the likelihood function, $\pi(p)$ is the prior distribution, and $\pi(x)$ is the marginal likelihood. The marginal likelihood is the probability of observing the data $x$. We could also express equation (3) as 

$$
f(\theta | data) = \frac{f(data | \theta)f(\theta)}{f(data)} \tag{4}
$$

where $f(\theta | data)$ is the posterior distribution, $f(data | \theta)$ is the likelihood function, $f(\theta)$ is the prior distribution. Notice that $f(data | \theta)$ is the _sampling density_ for the data - which is proportional to the likelihood function. The marginal likelihood $f(data)$ is the probability of observing the data $x$, which can be computed by integrating over all possible values of $\theta$

$$
f(data) = \int_{\theta} f(data | \theta)f(\theta) d\theta \tag{5}
$$

This number acts as a normalizing constant for the posterior distribution. Therefore, equation (4) can be rewritten as

$$
\begin{aligned}
f(\theta | data) & = \frac{f(data | \theta)f(\theta)}{f(data)} \\ 
& = \frac{f(data | \theta)f(\theta)}{\int_{\theta} f(data | \theta)f(\theta) d\theta} \tag{6}
\end{aligned}
$$

Therefore, Bayes' theorem is often stated as:

$$
\text{posterior} \varpropto \frac{\text{likelihood}\times\text{prior}}{\text{marginal likelihood}} \tag{7}
$$

## Conjugate prior

In the above example, we have a discrete prior distribution. However, we could also use a continuous prior distribution. For example, we could use a beta distribution to model the probability of success $p$.

In Bayesian statistics, a prior distribution is called a conjugate prior if the posterior distribution has the same functional form as the prior distribution. In other words, the posterior distribution is a member of the same family of distributions as the prior distribution.

For example, if we use a beta distribution to model the probability of success $p$, the posterior distribution will also be a beta distribution. In this case, the posterior distribution is a conjugate prior of the beta distribution.

How could choose a pirior distribution? We could use a conjugate prior if it is available. Otherwise, we could use a non-conjugate prior. In this case, we need to use a numerical method to compute the posterior distribution. But one of the most important rules is that we need to make sure that the domain of the random variable is within $[0,1]$.

<div class='figure'>
    <img src="/math/images/beta_distribution.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The plot of the beta distribution; notice that we are using lines to indicate that it is a continuous distribution.
    </div>
</div>

## Beta distribution

Now, we decide to use a beta distribution to model the probability of success $p$. The beta distribution is a continuous distribution with two parameters $\alpha$ and $\beta$. The probability density function of the beta distribution is given by

$$
f(p;\alpha,\beta) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)} \tag{8}
$$

where $B(\alpha,\beta)$ is the beta function. The beta function is defined as

$$
B(\alpha,\beta) = \int_{0}^{1} p^{\alpha-1}(1-p)^{\beta-1}dx = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} \tag{9}
$$

We can show that beta distribution is the conjugate prior of the binomial distribution. The posterior distribution is also a beta distribution. The posterior distribution is given by

$$
\begin{aligned}
\pi(p |x, \alpha, \beta) & = \frac{\pi(x|p)\pi(p|\alpha,\beta)}{\pi(x)} \\
                         & =\frac{\binom{n}{x}p^x(1-p)^{n-x}}{\pi(x)} \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)} \\
                         & = \binom{n}{x} \frac{p^{\alpha+x-1}(1-p)^{\beta+n-x-1}}{B(\alpha+x,\beta+n-x)} \\
                         & \varpropto p^{\alpha+x-1}(1-p)^{\beta+n-x-1} \tag{10}
\end{aligned}
$$

Therefore, $\alpha+x$ and $\beta+n-x$ are the new parameters of the posterior distribution. The posterior distribution is also a beta distribution.




## References

1. [Wikipedia: Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
2. [Conjugate pairs](https://drvalle1.github.io/9_Conjugate_pairs_example.html)





{% endkatexmm %}


