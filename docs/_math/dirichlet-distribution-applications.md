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
\begin{aligned}
\mathcal{f}(x \mid n, p) & =  \binom{n}{x_1} \binom{n-x_1}{x_2} \cdots \binom{n-x_1-x_2-\cdots-x_{k-1}}{x_k} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k} \\
& = \frac{n!}{x_1!(n-x_1)!} \frac{(n-x_1)!}{x_2!(n-x_1-x_2)!} \cdots  \\
&  \quad \quad \quad \frac{(n-x_1-x_2-\cdots-x_{k-1})!}{x_k!(n-x_1-x_2-\cdots-x_{k-1}-x_k)!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k} \\
& = \frac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k} \\
& = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k p_i^{x_i} \tag{8}
\end{aligned}
$$

where $n = x_1 + x_2 + \cdots + x_k$, and $p_1 + p_2 + \cdots + p_k = 1$.


As you can see in equation (8), we are using quite a lot of factorials. When it comes to factorials, we can leverage the Gamma function to simplify the expression. The Gamma function is defined as

$$
\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt \tag{9}
$$

where $t$ is a real number. Gamma function has a nice property which is given by the following equation:

$$
\Gamma(x+1) = x \Gamma(x) \tag{10}
$$

The proof of this equation is quite simple. We can use the following equation to prove it:

$$
\begin{aligned}
\Gamma(x+1) & = \int_0^\infty t^x e^{-t} dt \\
            & = \left [t^x (- e^{-t}) \right ]_0^\infty - \int_0^\infty (x t^{x-1} (-e^{-t})) dt \\
            & = (0 - 0) + x \int_0^\infty t^{x-1} e^{-t} dt \\
            & = x \Gamma(x) \tag{11}
\end{aligned}
$$

For every positive integer $n$, we have

$$
\begin{aligned}
\Gamma(n+1) & = n \Gamma(n) \\
            & = n (n-1) \cdots 2 \cdot 1 \\
            & = n! \tag{12}
\end{aligned}
$$

Equation (12) is recursive. 

The magic of Gamma function is that it can be used not only for positive integers, but also for real numbers. For example, we have

$$
\begin{aligned}
\Gamma(\frac{1}{2}) = \sqrt{\pi} \tag{13}
\end{aligned}
$$

To prove equation (13), we have to use the function of Gaussian distribution, which is given by

$$
\begin{aligned}
\mathcal{f}(x \mid \mu, \sigma^2) & = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2 \sigma^2}} \tag{14}
\end{aligned}
$$

where $\mu$ is the mean, and $\sigma^2$ is the variance. When $\mu = 0, \sigma^2 = 1/2$, we have

$$
\begin{aligned}
\mathcal{f}(x \mid 0, \frac{1}{2}) & = \frac{1}{\sqrt{ \pi}} e^{-x^2} \tag{15}
\end{aligned}
$$

Since Gaussian distribution is symmetric around $x=0$, we have

$$
\begin{aligned}
\int_{0}^\infty \mathcal{f}(x \mid 0, \frac{1}{2}) dx & = \int_{0}^\infty \frac{1}{\sqrt{\pi}} e^{-x^2} dx \\
& = \frac{1}{\sqrt{\pi}} \int_{0}^\infty e^{-x^2} dx \\
& = \frac{1}{2} \quad \text{(half of the area under the curve)} \tag{16}
\end{aligned}
$$

Therefore, we have

$$
\int_{0}^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2} \tag{17}
$$

With equation (17), we can prove equation (13) as follows:

$$
\begin{aligned}
\Gamma(\frac{1}{2}) & = \int_0^\infty t^{-\frac{1}{2}} e^{-t} dt \\
                    & = 2 \int_0^\infty e^{-u^2} du \\
                    & = \sqrt{\pi} \tag{18}
\end{aligned}                  
$$

We are using the substitution $u^2 = t$ in equation (18), which is valid because $u^2$ is always positive. Here is the process of substitution:

$$
\begin{aligned}
u^2 = t & \Rightarrow u = t^{\frac{1}{2}} \\
\frac{du}{dt}  = \frac{1}{2} t^{-\frac{1}{2}} & \Rightarrow 2 du = t^{-\frac{1}{2}} dt \\
\end{aligned}
$$

## Jacobian matrix 

In the last section, we said that we will deal with multi-category multinomial distribution. We state that we have $k$ categories, and each category has a probability $p_i$, where $i \in \{1, 2, \cdots, k\}$. For different categories, we have a vector $x = {x_1, x_2, \cdots, x_k}$, where $x_i$ is the number of samples in category $i$. 

From data generating process, we can model different stages of this process. For instance,

- we can model whether a category will show up or not in $n$ trials, and use one-hot encoding to represent the result, such as $x = [0, 1, 0, 0, 0]$, which means that the second category shows up in $n$ trials, and the rest of the categories do not show up in $n$ trials.
- we can also model how many category will show up in $n$ trials, such as $x = [0, 1, 0, 0, 1]$, which means that the second category and the fifth category show up in $n$ trials, and the rest of the categories do not show up in $n$ trials.
- we can also model the number of samples in each category, such as $x = [1, 6, 3, 4, 5]$, which means that the first category shows up once, the second category shows up six times, and so on.

For each stage, no matter how we model the data, we can use a regression to link each element of the vector $x$ to a series of independent variables, such as 

$$
\begin{aligned}
x_1 & = \Phi _1(y_1, y_2, \cdots, y_m) \\
x_2 & = \Phi_2(y_1, y_2, \cdots, y_m) \\
\vdots & \quad \quad \quad \quad \quad  \vdots \\
x_k & = \Phi_k(y_1, y_2, \cdots, y_m) \\ \tag{19}
\end{aligned}
$$

where $\Phi_i$ is a regression function, and $y_1, y_2, \cdots, y_m$ are independent variables. If we want to use machine learning or deep learning to learn the regression function $\Phi_i$, we need to know the Jacobian matrix of $\Phi_i$.

The Jacobian matrix of $\Phi_i$ is given by

$$
\frac{\partial (x_1, x_2 \cdots x_k) }{\partial (y_1, y_2, \cdots, y_m)} = \begin{bmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & \cdots & \frac{\partial x_1}{\partial y_m} \\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} & \cdots & \frac{\partial x_2}{\partial y_m} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial x_k}{\partial y_1} & \frac{\partial x_k}{\partial y_2} & \cdots & \frac{\partial x_k}{\partial y_m} \\
\end{bmatrix} \tag{20}
$$

where we use the chain rule in equation (20).

Now, to make our life easier, we will set $m = k$, which means we have $k$ independent variables, and $k$ regression functions. In this case, we can use the following equation to calculate the Jacobian matrix:

$$
\frac{\partial (x_1, x_2 \cdots x_k) }{\partial (y_1, y_2, \cdots, y_k)} = \begin{bmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & \cdots & \frac{\partial x_1}{\partial y_k} \\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} & \cdots & \frac{\partial x_2}{\partial y_k} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial x_k}{\partial y_1} & \frac{\partial x_k}{\partial y_2} & \cdots & \frac{\partial x_k}{\partial y_k} \\
\end{bmatrix} \tag{21}
$$

The good thing about using the same dimension for $m$ and $k$ is that we preserve structure of the linear space as they have the same number of coordinates. Then, the intuitive meaning of integral (whether it is area or volume) is preserved but stretched or compressed with the determinant of the Jacobian matrix.

We denote $J$ as the determinant of the Jacobian matrix, which is given by

$$
J =  \left| \frac{\partial (x_1, x_2 \cdots x_k) }{\partial (y_1, y_2, \cdots, y_k)} \right| \tag{22}
$$

Now, when we map a region $D$ in $k$-dimensional space to another region $D'$ in $k$-dimensional space, we can use the following equation to calculate the volume of $D'$:

$$
\int \dotsi_{D} \int f(x_1, x_2, \cdots, x_k) dx_1 dx_2 \cdots dx_k = \int \dotsi_{D'} \int f(y_1, y_2, \cdots, y_k) J dy_1 dy_2 \cdots dy_k\tag{23}
$$

For those who want to refresh their memory about the Jacobian matrix, please refer to the following documents:[Jacobian](../../pdf/jacobian_examples.pdf){:target="_blank"}.



## From Chi-square distribution to Gamma distribution to Beta distribution

A random variable $X$ has a chi-square distribution with $k$ degrees of freedom if its probability density function is given by

$$
X = Y_1^2 + Y_2^2 + \cdots + Y_k^2 \tag{24}
$$

where $Y_1, Y_2, \cdots, Y_k$ are independent and identically distributed random variables with a _standard normal distribution_.

The Gamma distribution is a generalization of the chi-square distribution.
If a random variable $Z$ has a Chi-square distribution with $k$ degrees of freedom, and $\theta$ is a positive constant, then the random variable $X$ defined by

$$
X = \frac{\theta }{k} Z \tag{25}
$$

has a Gamma distribution with shape parameter $k$ and scale parameter $\theta$. We often use $\alpha$ to denote the shape parameter, and $\beta$ to denote the scale parameter. Then, probability density function of the Gamma distribution is given by

$$
f(x) = \begin{cases}
\frac{1}{\Gamma(\alpha) \beta^\alpha} x^{\alpha - 1} e^{-\frac{x}{\beta}} & x > 0 \\
0 & x \leq 0
\end{cases} \tag{26}
$$

where $\alpha > 0, \beta > 0, \Gamma(\alpha)$ is the gamma function, which is given by

$$
\Gamma(\alpha) = \int_0^\infty t^{\alpha - 1} e^{-t} dt \tag{27}
$$

The Beta distribution is a generalization of the Gamma distribution. If a random variable $X \sim \mathrm{Gamma}(\alpha, 1)$ and $Y \sim \mathrm{Gamma}(\beta, 1)$, then the random variable $Z = \frac{X}{X + Y}$ has a Beta distribution with parameters $\alpha$ and $\beta$:

$$
\frac{X}{X + Y} \sim \mathrm{Beta}(\alpha, \beta) \tag{28}
$$

The probability density function of the Beta distribution is given by

$$
f(x) = \begin{cases}
\frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1} & 0 < x < 1 \\
0 & \text{otherwise}
\end{cases} \tag{29}
$$

where $B(\alpha, \beta)$ is the beta function, which is given by

$$
B(\alpha, \beta) = \int_0^1 t^{\alpha - 1} (1 - t)^{\beta - 1} dt \tag{30}
$$


We could also express equation (30) as

$$
B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)} \tag{31}
$$

To derive equation (31), we will first show 

$$
\begin{aligned}
& \int_0^1 f(x) dx = 1 = \int_0^1 \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1} dx \\ 
& \Rightarrow B(\alpha, \beta) = \int_0^1 x^{\alpha - 1} (1 - x)^{\beta - 1} dx \quad \text{same as equation (30)} \\
\end{aligned} \tag{32}
$$

Now, let's calculate the value of $\Gamma(\alpha) \Gamma(\beta)$:

$$
\begin{aligned}
\Gamma(\alpha) \Gamma(\beta) & = \int_0^\infty u^{(\alpha - 1)} e^{-u} du \int_0^\infty v^{(\beta - 1)} e^{-v} dv \\
& = \int_0^\infty \int_0^\infty u^{(\alpha - 1)} v^{(\beta - 1)} e^{-(u + v)} du dv 
\end{aligned}
$$

Now, we set $x = \frac{u}{u + v}$ and $y = u+v$, with the bounds $0 \leq x \leq 1$ and $0 \leq y \leq \infty$. Then, we have the mapping from $uv$ to $xy$:

$$
u = xy, \quad v = (1-x)y \tag{33}
$$

The Jacobian matrix is given by

$$
\frac{\partial (u, v)}{\partial (x, y)} = 
\begin{bmatrix}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
\end{bmatrix} = 
\begin{bmatrix}
y & x \\
-y & 1 - x
\end{bmatrix} \tag{34}
$$

The jacobian is then given by

$$
J = \left| \frac{\partial (u, v)}{\partial (x, y)} \right|  = y(1-x) - x(-y) = y \tag{35}
$$

By transforming the integral, we have

$$
\begin{aligned}
\Gamma(\alpha) \Gamma(\beta) & = \int_{y=0}^\infty \int_{x=0}^1 (xy)^{(\alpha - 1)} [(1 - x)y]^{(\beta - 1)} e^{-y} J dx dy \\
& = \int_{y=0}^\infty y^{(\alpha + \beta - 1)} e^{-y} dy \int_{x=0}^1 x^{\alpha - 1} (1 - x)^{\beta - 1} dx \\
& = \Gamma(\alpha + \beta) \int_{x=0}^1 x^{\alpha - 1} (1 - x)^{\beta - 1} dx \\
& = \Gamma(\alpha + \beta) B(\alpha, \beta) 
\end{aligned} \tag{36}
$$

This concludes the proof of equation (31).











{% endkatexmm %}
