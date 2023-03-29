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
                         & \varpropto p^{\alpha+x-1}(1-p)^{\beta+n-x-1} \tag{10}
\end{aligned}
$$

Note: the full derivation of the posterior distribution is beyond the scope of this course. If you are interested in the derivation, you could refer to the following resources: [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior){target="_blank"}.

Therefore, $\alpha+x$ and $\beta+n-x$ are the new parameters of the posterior distribution. The posterior distribution is also a beta distribution.

Since the beta distribution has two parameters, we need to specify the values of $\alpha$ and $\beta$. We could use the following rules to choose the values of $\alpha$ and $\beta$.

1. If we have no prior knowledge about the probability of success $p$, we could set $\alpha=\beta=1$. In this case, the prior distribution is uniform.
2. If we have a strong belief that the probability of success $p$ is close to 0, we could set $\alpha=1$ and $\beta=100$. In this case, the prior distribution is concentrated around 0.
3. If we have a strong belief that the probability of success $p$ is close to 1, we could set $\alpha=100$ and $\beta=1$. In this case, the prior distribution is concentrated around 1.

Those rules just give some examples. We could use other values for $\alpha$ and $\beta$. In practice, it is better to combine our prior knowledge with the data to choose the values of $\alpha$ and $\beta$.

For instance, in our example, we could ask two questions:

1. What is the average number of times this player would score out of 1,000 shots?
2. What is the value Z for which we believe that it is extremely unlikely that this player will score less than Z out of 1,000 shots?

The first question is equivalent to asking what is the probability of success $p$. The second question is equivalent to asking what is the value of $p$ such that the probability of success is less than 0.01. 

We will transfer our prior knowledge into statistical terms:

$$
\begin{aligned}
& \mathbb{E}[\pi] = \mu = \frac{500}{1000} \\ 
& \mathbb{P}[\pi < 0.01] = 0.01 \\
\end{aligned}
$$

with those two values, we could try different values for $\alpha$ and $\beta$ to find the best values. By the expectation of beta distribution, we could find that

$$
\begin{aligned}
\mathbb{E}[\pi] & = \frac{\alpha}{\alpha+\beta} = \frac{500}{1000} = 0.5  \\
\Rightarrow \beta & = \frac{alpha (1-\mu)}{\mu} 
\end{aligned}
$$

To calculate the probability in equation (8), we also need set up a prior for $p$. The following R code shows how to calculate the probability of success $p$.

<div class='figure'>
    <img src="/math/images/beta_prior.png"
         alt="beta prior"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The plot of the beta distribution based on the prior knowledge for different values of $\alpha$ and $\beta$; notice that the shape is same because we set the expectation equal to 0.5.
    </div>
</div>


```R
# find values of alpha and beta that satify the following conditions:
# E[p] = 0.5
# P(p < 0.01) = 0.01

# prior probability of success
# the value can be changed based on the prior knowledge
# in practice, we set up alpha and beta and search for 
# the value of p that satisfies the conditions
prior_p = 100/1000 
prior_mean = 500/1000

# generate a grid of alpha and beta
alpha = seq(0.001, 10, length.out = 1000)
beta = alpha * (1 - prior_mean) / prior_mean

# calcuate the prior probability based on beta distribution
prior = pbeta(prior_p, alpha, beta)

# plot the prior probability for alpha and beta in two plots and
# combine them into one plot

png("../math/images/beta_prior.png", width = 7, height = 5,
                units = "in", res = 300)
par(mfrow = c(1, 2))
plot(alpha, prior, type = "l", xlab = "alpha",
            ylab = "prior probability", 
            main = "Prior Probability - alpha")
abline(h=0.01, col="red", lty=2)

plot(beta, prior, type = "l", xlab = "beta",
            ylab = "prior probability", 
            main = "Prior Probability - beta")
abline(h=0.01, col="red", lty=2)
dev.off()

# get alpha and beta that satisfy the conditions
# calculate the probability that is close to 0.01
p_dist = abs(prior-0.01)
idx = which(p_dist == min(p_dist))
alpha[idx]  # 2.863
beta[idx]  # 2.863

# plot the beta distribution for the values of alpha and beta
png("../math/images/beta_prior2.png", width = 6, height = 4,
                units = "in", res = 300)
p_seq = seq(0, 1, length.out = 1000)
plot(p_seq, dbeta(p_seq, alpha[idx], beta[idx]), type = "l", 
    xlab = "p", ylab = "density", 
    main = "Beta Distribution - alpha = 2.863, beta = 2.863")
abline(v = prior_p, col = "red", lty = 2)
abline(v = prior_mean, col="gray", lty=2)
dev.off()
```

<div class='figure'>
    <img src="/math/images/beta_prior2.png"
         alt="beta prior"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The plot of the beta distribution, which has the bell shape. 
    </div>
</div>


If you read the code and check Figure 3, you should realize that we are picking up values of $\alpha$ and $\beta$ based on the prior knowledge. Those prior knowledge set up two constraints for the values of $\alpha$ and $\beta$. The first constraint is that the expectation of the beta distribution is 0.5. The second constraint is that the probability of success $p$ is less than 0.01 should be rare.

## Posterior distribution

We have the prior distribution for the probability of success $p$. Now we have the data, which is the number of shots and the number of goals. We could use the data to update the prior distribution to get the posterior distribution. The posterior distribution is the distribution of the probability of success $p$ after we have the data.

In the section 2 - prior belief, we have the following values for prior probability of success $p$:

$$
\begin{aligned}
\pi(p = 0.7) & = 0.2 \\
\pi(p = 0.5) & = 0.75 \\
\pi(p = 0.1) & = 0.05 \\
\end{aligned}
$$

Now, our basketball player is going to shoot. She/he scores 3 goals out of 10 shots. How could we update the prior distribution based on the data? We could use the following formula to calculate the posterior distribution:

$$
\begin{aligned}
f(\theta | data) & = \frac{f(data | \theta) f(\theta)}{f(data)} \\
\end{aligned}
$$

We are using $f$ because they are referring to the probability density function. The following table shows the calculation of the posterior distribution.

| category | binom_p| priro | likelihood| posterior|
|------:|:-------:|:-----:|:----------:|:---------:|
|good |     0.7|  0.20|      0.009|     0.019|
|average |     0.5|  0.75|      0.117|     0.950|
|bad |     0.1|  0.05|      0.057|     0.031|

After seeing the posterior, we come to the conclusion this basketball player is not a very good shooter as the weight of average player increases. 

The following R code shows how to calculate the posterior distribution.

```R
# calculate the posterior probability of success
# x = number of goals
# n = number of shots
x = 3
n = 10
# prior probability of success for three categories
# good player, average player, bad player
binom_p = c(0.7, 0.5, 0.1)
prior = c(0.2, 0.75, 0.05)
# calculate the likelihood of success
likelihood = dbinom(x, n, binom_p)

# 0.0090016920.11718750.057395628
# each category has a different likelihood of success

# calculate the posterior probability of success
posterior = likelihood * prior / sum(likelihood * prior)

# create a data frame to store the results
df = data.frame(binom_p, prior, likelihood, posterior)
names(df) = c("binom_p", "prior", "likelihood", "posterior")

df %>%
    round(3) %>%
    kable("markdown", align = "c")
```

Since we are using discrete values for the probability of success $p$, we could use the following formula to calculate the posterior distribution:

$$
\begin{aligned}
\pi(\theta | data) & = \frac{\pi(data | p) \pi(p)}{\pi(data)} \\
                & = \frac{\pi(data | p) \pi(p)}{\sum_{i} \pi(data | p= i) \pi(p = i)} \\
\end{aligned}
$$


Now, instead of using the discrete values for the probability of success $p$, we could use the continuous values for the probability of success $p$. 

In equation (10), we have already derived the formula for the posterior distribution. We could use the following formula to calculate the posterior distribution:

$$
\pi(p | x, \alpha, \beta) \varpropto p^{x + \alpha - 1} (1 - p)^{n - x + \beta - 1}
$$

Therefore, the posterior distribution for this basketball player is given by the following formula:

$$
\pi(p | x, \alpha, \beta) \varpropto p^{3 + 2.863-1} (1 - p)^{10 - 3 + 2.863 - 1} = \Beta (3+2.863, 10-3+2.863)
$$


<div class='figure'>
    <img src="/math/images/beta_prior3.png"
         alt="beta prior"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The plot of the prior and posterior distributions. Notice how the posterior distribution is shifted to the left. 
    </div>
</div>

Figure 4 shows that the posterior distribution has shifted to the left. The posterior distribution is more concentrated on the left side, which indicates that the basketball player is not a very good shooter.

Here is the R code to plot the prior and posterior distributions.

```R
# plot the prior and posterior distributions
x = 3
n = 10
p_seq = seq(0, 1, length.out = 1000)
plot(p_seq, dbeta(p_seq, alpha[idx], beta[idx]), type = "l", 
    xlab = "p", ylab = "density", ylim = c(0, 4),
    main = "Update Beta Distribution")
lines(p_seq, dbeta(p_seq, alpha[idx] + x, beta[idx] + n - x), 
    col = "red", lwd = 2)
legend("topright", legend = c("prior", "posterior"), 
    col = c("black", "red"), lty = 1, lwd = 2)
```



## References

1. [Wikipedia: Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
2. [Conjugate pairs](https://drvalle1.github.io/9_Conjugate_pairs_example.html)





{% endkatexmm %}


