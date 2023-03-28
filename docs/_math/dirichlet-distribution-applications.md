---
title: Dirichlet Distribution and Its Applications
subtitle: From latent Dirichlet allocation to Bayesian inference, and beyond, the Dirichlet distribution is a powerful tool in the data scientist's toolbox.
layout: math_page_template
date: 2023-03-28
keywords: probabilistic-thinking dirichlet-distribution text-mining machine-learning bayesian-inference bayesian-statistics 
published: true
tags: probability algorithm data-science machine-learning dirichlet-distribution high-dimensional-data
---

It has been twenty years since the paper Latent Dirichlet Allocation was published by {% cite blei2003latent %}. The paper introduced a new model for text mining, and it has since become one of the most popular models for text mining. The model is based on the Dirichlet distribution, which is a distribution over probability distributions. The Dirichlet distribution is a powerful tool in the data scientist's toolbox, and it is used in a variety of applications, including Bayesian inference, text mining, and high-dimensional data analysis.

The original idea of using the Dirichlet distribution for inference of population structure was introduced in the context of population genetics by {% cite pritchard2000inference %}. 

Both papers are great and worth reading, but they are not easy to understand. To be fair, the paper by {% cite pritchard2000inference %} is easier to follow than the paper by {% cite blei2003latent %}.

In this post, I will try to explain the Dirichlet distribution in a simple way, and I will also discuss some of its applications.

## From bernoulli to binomial to poisson

{% katexmm %}

before we can discuss the Dirichlet distribution, we need to discuss the concept of a probability distribution. A probability distribution is a function that assigns a probability to each possible outcome of a random experiment. 

For example, the Bernoulli distribution is a probability distribution that assigns a probability to the outcome of a coin flip. The Bernoulli distribution has a single parameter, which is the probability of getting a head. The Bernoulli distribution is defined as follows (probability mass function):

$$
\mathcal{f}(x \mid p) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\} \tag{1}
$$

where $x$ is the outcome of the coin flip, and $p$ is the probability of getting a head. The Bernoulli distribution is a discrete probability distribution, which means that it assigns a probability to each possible outcome of the coin flip. In this case, the possible outcomes are 0 and 1, and the probability of getting a head is $p$. The probability of getting a tail is $1-p$.


Now if we run the coin flip experiment multiple times, we are conducting binomial experiment. The binomial experiment is a random experiment that consists of $n$ coin flips. The binomial experiment has two parameters: $n$ and $p$. The binomial distribution (probability mass function) is defined as follows:

$$
\mathcal{f}(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}, \tag{2}
$$

where $x$ is the number of heads in $n$ coin flips, and $p$ is the probability of getting a head. The binomial distribution is a discrete probability distribution, which means that it assigns a probability to each possible outcome of the binomial experiment. In this case, the possible outcomes are the number of heads in $n$ coin flips, and the probability of getting $x$ heads is given by the binomial distribution.

Figure 1 gives the plot of the binomial distribution for different values of $n$ and $p$.


<div class='figure'>
    <img src="/math/images/binomial_distribution.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The plot of the binomial distribution for different values of $n$ and $p$. Notice that we are using different markers and connected lines to emphasize it is a discrete probability distribution.
    </div>
</div>


The connection between the Bernoulli distribution and the binomial distribution is that the binomial distribution is the distribution of the number of heads in $n$ coin flips, where each coin flip is a Bernoulli trial. And each trial is run independently of the other trials.


At this stage, you need to understand the concept of identically independent (i.i.d) random variables. Identically independent random variables are random variables that are independent of each other, and they are also identically distributed. In other words, the random variables are independent of each other, and they have the same distribution.


In this case, we say that the coin flips are identically independent Bernoulli trials. This means that each coin flip is independent of the other coin flips, and each coin flip is a Bernoulli trial.


Binomial distribution is connected with the Poisson distribution. The Poisson distribution is a discrete probability distribution that assigns a probability to the number of events that occur in a fixed interval of time. The Poisson distribution has a single parameter, which is the average number of events that occur in a fixed interval of time. The Poisson distribution is defined as follows (probability mass function):

$$
\mathcal{f}(x \mid \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x \in \mathbb{N} \tag{3}
$$

where $x$ is the number of events that occur in a fixed interval of time, and $\lambda$ is the average number of events that occur in a fixed interval of time. The Poisson distribution is a discrete probability distribution, which means that it assigns a probability to each possible outcome of the Poisson experiment. In this case, the possible outcomes are the number of events that occur in a fixed interval of time, and the probability of getting $x$ events is given by the Poisson distribution.

Figure 2 gives the plot of the Poisson distribution for different values of $\lambda$.

<div class='figure'>
    <img src="/math/images/poisson_distribution.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The plot of the Poisson distribution for different $\lambda$. Notice that we are using different markers and connected lines to emphasize it is a discrete probability distribution.
    </div>
</div>

Now, we will derive the Poisson distribution from the binomial distribution. To do this, we need to calculate the expectation of the binomial distribution (see the [derivation](https://proofwiki.org/wiki/Expectation_of_Binomial_Distribution){:target="_blank"}). The expectation of the binomial distribution is given by:

$$
\mathbb{E}(X) = np \tag{4}
$$

If you check the figure 1 again, you should notice that the expectation of the binomial distribution shifts to the right as $n$ increases or $p$ increases. This is because the expectation of the binomial distribution is the number of heads in $n$ coin flips, and the number of heads increases as $n$ increases or $p$ increases.

Now, we let $\lambda = np$. Then, we can write the probability mass function of the binomial distribution as follows:

$$
\begin{aligned}
\mathcal{f}(x \mid n, p) & = \binom{n}{x} p^x (1-p)^{n-x} \\
& = \frac{n!}{x!(n-x)!} p^x (1-p)^{n-x} \\
& = \frac{n!}{x!(n-x)!} \left(\frac{\lambda}{n}\right)^x \left(1-\frac{\lambda}{n}\right)^{n-x} \\
& = \frac{n!}{x!(n-x)!} \frac{\lambda^x}{n^x} \left(1-\frac{\lambda}{n}\right)^{n} \left(1-\frac{\lambda}{n}\right)^{-x} \\
& = \frac{\lambda^x}{x!} \frac{n!}{(n-x)! n^x } \left(1-\frac{\lambda}{n}\right)^{n} \left(1-\frac{\lambda}{n}\right)^{-x} \\ \tag{5}
\end{aligned}
$$

For the last three components of equation (5), we have the following:

$$
\begin{aligned}
\lim_{n \to \infty} \frac{n !}{(n-x)! n^x} & = \lim_{n \to \infty} \frac{n !}{(n-x)! n^x} \\ 
& = \lim_{n \to \infty} \frac{n (n-1) \cdots (n-x+1)}{n^x}  \\
& = \lim_{n \to \infty} \frac{O(n^x)}{O(n^x)} \quad \text{there are x terms} \\ 
& = 1 \\
\lim_{n \to \infty} \left(1-\frac{\lambda}{n}\right)^{n} & = e^{-\lambda}\\
\lim_{n \to \infty} \left(1-\frac{\lambda}{n}\right)^{-x} & = 1 \tag{6}
\end{aligned}
$$

Therefore, we have

$$
\begin{aligned}
\mathcal{f}(x \mid n, p) & = \frac{\lambda^x}{x!} e^{-\lambda} \\ \tag{7}
\end{aligned}
$$

The intuition behind this derivation is that we fix the expectation of binomial distribution to be $\lambda$, and then we run $n \to \infty$ to check within a fixed interval of time, what's the probability of getting $x$ events. This is exactly the definition of the Poisson distribution.


## Multnomial distribution

The multinomial distribution is a generalization of the binomial distribution. Instead of having two possible outcomes, the multinomial distribution has $k$ possible outcomes with probabilities $p_1, p_2, \cdots, p_k$ for each outcome. The number of successes for each outcome is $x_1, x_2, \cdots, x_k$, respectively. 

Let's have an example. Suppose we will run $n$ trials and each trial could
have three possible outcomes. Here is the key points of this example:

- $n$ trials
- the possible outcomes are: $A$, $B$, and $C$
- The probability of getting
those outcomes correspondingly are: $p_A$, $p_B$, and $p_C$.
- in $n$ trials, we could have all $A$'s, all $B$'s, all $C$'s, or a combination of $A$'s, $B$'s, and $C$'s.
- this is then a problem of permutation (order matters)

Now, let vector $x = (x_1, x_2, \cdots, x_k)$ be the number of successes for each outcome, and $p = (p_1, p_2, \cdots, p_k)$. Then, the probability of getting $x$ is given by the multinomial distribution:

$$
\being{aligned}
\mathcal{f}(x \mid n, p) & =  \binom{n}{x_1} \binom{n-x_1}{x_2} \cdots \binom{n-x_1-x_2-\cdots-x_{k-1}}{x_k} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k} \\
\frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k} 
\tag{8}
\end{aligned}
$$

where $n = x_1 + x_2 + \cdots + x_k$, and $p_1 + p_2 + \cdots + p_k = 1$.






Since we have multiple outcomes, it is natural to model the multinomial distribution as a vector of probabilities. 







{% endkatexmm %}
