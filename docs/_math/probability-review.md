---
title: Probability Review
subtitle: From time to time, we need to review those definitions and basic concepts from probability field.
layout: math_page_template
date: 2023-01-03
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality 
published: true
tags: probability  data-science machine-learning
---


This post is to go through a review of probability concepts whenever you need it. 

{% katexmm %}

To treat probability rigorously, we define a _sample space_ $S$ whose elements 
are the possible outcomes of some process or experiment. For example, 
the sample space might be the outcomes of the roll
of a die, or flips of a coin. To each element $x$ of the sample space,
 we assign a probability, which
will be a non-negative number between 0 and 1, which we will denote by $\mathbb{P}(x)$.
We require that

$$\sum_{x \in S} \mathbb{P}(x) = 1$$


Let $A$ be an event with non-zero probability. We define the probability of an event $B$ conditioned
on event A, denoted by $\mathbb{P}(B | A)$, to be


$$\mathbb{P}(B | A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)}$$


If we have two events $A$ and $B$, we say that they are independent if the probability that both
happen is the product of the probability that the first happens and the probability that the second
happens, that is, if

$$\mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$ 

__Bayes's rule.__ For any two events $A$ and $B$, one has

$$\mathbb{P}(B | A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)} =   \frac{\mathbb{P}(A| B) \mathbb{P}(B)}{\mathbb{P}(A)} $$


Suppose we have a false positive rate:

$$\mathbb{P}(\text{positive test} | \text{no disease}) = \frac{1}{30}$$ 

and some false negative rate is 

$$\mathbb{P}(\text{negative test} | \text{ disease}) = \frac{1}{10}$$ 

The incidence of the desease is $1/1000$

$$\mathbb{P}(B) = 0.001$$


Let's define:

- $A$: testing positive
- $B$: having the disease 

Then we can have

$$
\begin{aligned}
\mathbb{P}(B | A) = \frac{\mathbb{P}( A \cap B)}{\mathbb{P}(A)} & = \frac{\mathbb{P}(A | B) \mathbb{P}(B) }{\mathbb{P}(A)} \\ 
& = \frac{\mathbb{P}(A | B) \mathbb{P}(B) }{\mathbb{P}(A|B)\mathbb{P}(B) + \mathbb{P}(A | \neg B )\mathbb{P}(\neg B)}  \\
& \approx 0.0265
\end{aligned}
$$


__Inclusion–exclusion principle.__ In combinatorics, a branch of mathematics, 
the inclusion–exclusion principle is a counting technique which generalizes the 
familiar method of obtaining the number of elements in the union of two finite 
sets; symbolically expressed as

$$|A \cup B| = |A| + |B| - |A \cap B |$$

The inclusion-exclusion principle, being a generalization of the two-set case, is perhaps more clearly seen in the case of three sets, which for the sets A, B and C is given by

$$ 
\begin{aligned}
|A\cup B\cup C|=|A|+|B|+|C|- & |A\cap B|-  |A\cap C| \\
    & -|B\cap C|+|A\cap B\cap C|
\end{aligned}
$$

<div class='figure'>
    <img src="/images/math/inclusion-exclusion.png"
         alt="A demo figure"
         style="width: 40%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Illustration of inclusion-exclusion principle. 
    </div>
</div>

In its general formula, the principle of inclusion–exclusion states that for finite sets A1, …, An, one has the identity

$$ 
\begin{aligned}
\left|\bigcup _{i=1}^{n}A_{i}\right|=\sum _{i=1}^{n}|A_{i}|-\sum _{1\leqslant i<j\leqslant n}|A_{i}\cap A_{j}|+ & \sum _{1\leqslant i<j<k\leqslant n}|A_{i}\cap A_{j}\cap A_{k}|- \\
& \cdots +(-1)^{n+1}\left|A_{1}\cap \cdots \cap A_{n}\right| 
\end{aligned}
$$

__Expectation.__ Informally, the expectation of a random variable with a countable set of 
possible outcomes is defined analogously as the weighted average of all 
possible outcomes, where the weights are given by the probabilities of 
realizing each given value. This is to say that

$$ \operatorname {E} [X]=\sum _{i=1}^{\infty }x_{i}\,p_{i},$$

where $x_1, x_2, \cdots$ are the possible outcomes of the random variable 
$X$ and $p_1, p_2, \cdots$ are their corresponding probabilities.

Now consider a random variable $X$ which has a probability density function 
given by a function f on the real number line. This means that the 
probability of $X$ taking on a value in any given open interval is given by 
the integral of $f$ over that interval. The expectation of $X$ is then given 
by the integral

$$ \operatorname {E} [X]=\int _{-\infty }^{\infty }x \ f(x)\,dx $$

__Indicator function.__ The indicator variable(function) of a subset $A$ of a set $X$ is a function

$$ \mathbf{I} _{A}\colon X\to \{0,1\} $$

defined as

$$ \mathbf {I} _{A}(x):={\begin{cases}1~&{\text{ if }}~x\in A~,\\0~&{\text{ if }}~x\notin A~.\end{cases}} $$

At first glance, there might not seem like much point in using such a simple random variable.
However, it can be very _useful_, especially in conjunction with the following important fact.


__Linearity of expectation.__ The expected value operator (or expectation operator)
$ \operatorname {E} [\cdot ]$ is linear in the sense that, for any random variables
$X$ and $Y$, and a constant $a$,

$$ {\begin{aligned}\operatorname {E} [X+Y]&=\operatorname {E} [X]+\operatorname {E} [Y],\\\operatorname {E} [aX]&=a\operatorname {E} [X],\end{aligned}}$$

whenever the right-hand side is well-defined. When $X$ and $Y$ are independent, we
also have 

$$E(X \cdot Y) = E(X) E(Y)$$

If $X$ and $Y$ are independent, it holds that

$$p(x,y) = p(x) \, p(y) \quad \text{for all} \quad x \in \mathcal{X}, y \in \mathcal{Y}$$

Applying this to the expected value for discrete random variables, we have

$$
\begin{aligned}
\mathrm{E}(X\,Y) &= \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} (x \cdot y) \cdot f_{X,Y}(x,y) \\
&= \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} (x \cdot y) \cdot \left( f_X(x) \cdot f_Y(y) \right) \\
&= \sum_{x \in \mathcal{X}} x \cdot f_X(x) \, \sum_{y \in \mathcal{Y}} y \cdot f_Y(y) \\
&= \sum_{x \in \mathcal{X}} x \cdot f_X(x) \cdot \mathrm{E}(Y) \\
&= \mathrm{E}(X) \, \mathrm{E}(Y)
\end{aligned}
$$

And applying it to the expected value for continuous random variables, we have

$$
\begin{aligned}
\mathrm{E}(X\,Y) &= \int_{\mathcal{X}} \int_{\mathcal{Y}} (x \cdot y) \cdot f_{X,Y}(x,y) \, \mathrm{d}y \, \mathrm{d}x \\
&= \int_{\mathcal{X}} \int_{\mathcal{Y}} (x \cdot y) \cdot \left( f_X(x) \cdot f_Y(y) \right) \, \mathrm{d}y \, \mathrm{d}x \\
&= \int_{\mathcal{X}} x \cdot f_X(x) \, \int_{\mathcal{Y}} y \cdot f_Y(y)  \, \mathrm{d}y \, \mathrm{d}x \\
&= \int_{\mathcal{X}} x \cdot f_X(x) \cdot \mathrm{E}(Y) \, \mathrm{d}x \\
&= \mathrm{E}(X) \, \mathrm{E}(Y) \; 
\end{aligned}
$$


__Variance and Covariance.__ The variance of a random variable $X$ is the expected value of the squared deviation 
from the mean of $X$, $\mu =\operatorname {E} [X]$:

$$ 
\operatorname {Var} (X)=\operatorname {E} \left[(X-\mu )^{2}\right].
$$

This definition encompasses random variables that are generated by processes that are 
discrete, continuous, neither, or mixed. The variance can also be thought of as the covariance of a random variable with itself:

$$ \operatorname {Var} (X)=\operatorname {Cov} (X,X).$$

For two jointly distributed real-valued random variables $X$ and $Y$ with finite second 
moments, the covariance is defined as the expected value (or mean) of the 
product of their deviations from their individual expected values:

$$ 
\operatorname {cov} (X,Y)=\operatorname {E} {\big [(X-\operatorname {E} [X])(Y-\operatorname {E} [Y])\big ]}
$$

where $ \operatorname {E} [X]$ is the expected value of $X$, also known as the mean of 
$X$. The covariance is also sometimes denoted $\sigma _{XY}$ or $\sigma (X,Y)$, in analogy to variance. 


The expression for the variance can be expanded as follows:

$$
\begin{aligned}\operatorname {Var} (X)&=\operatorname {E} \left[(X-\operatorname {E} [X])^{2}\right]\\[4pt]&=\operatorname {E} \left[X^{2}-2X\operatorname {E} [X]+\operatorname {E} [X]^{2}\right]\\[4pt]&=\operatorname {E} \left[X^{2}\right]-2\operatorname {E} [X]\operatorname {E} [X]+\operatorname {E} [X]^{2}\\[4pt]&=\operatorname {E} \left[X^{2}\right]-\operatorname {E} [X]^{2}\end{aligned}
$$

In other words, the variance of X is equal to the mean of the square of X minus the square of the mean of X. This equation should not be used for computations using floating point arithmetic, because it suffers from catastrophic cancellation if the two components of the equation are similar in magnitude.


For the covariance, by using the linearity property of expectations, it can be simplified to the expected value of their product minus the product of their expected values:

$$
\begin{aligned}\operatorname {cov} (X,Y)&=\operatorname {E} \left[\left(X-\operatorname {E} \left[X\right]\right)\left(Y-\operatorname {E} \left[Y\right]\right)\right]\\&=\operatorname {E} \left[XY-X\operatorname {E} \left[Y\right]-\operatorname {E} \left[X\right]Y+\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]\right]\\&=\operatorname {E} \left[XY\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]+\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]\\&=\operatorname {E} \left[XY\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]
\end{aligned}
$$

but this equation is susceptible to catastrophic cancellation. 


__Covariance of independent random variables.__  Let $X$ and $Y$ 
be independent random variables. Then, the covariance of $X$ and $Y$ 
is zero:

$$X, Y \; \text{independent} \quad \Rightarrow \quad \mathrm{Cov}(X,Y) = 0 $$

The covariance can be expressed in terms of expected values as

$$\mathrm{Cov}(X,Y) = \mathrm{E}(X\,Y) - \mathrm{E}(X) \, \mathrm{E}(Y) $$

For independent random variables, the expected value of the product is equal to the product of the expected values:

$$\mathrm{E}(X\,Y) = \mathrm{E}(X) \, \mathrm{E}(Y)$$

Therefore, 

$$
\begin{aligned}
\mathrm{Cov}(X,Y) &= \mathrm{E}(X\,Y) - \mathrm{E}(X) \, \mathrm{E}(Y) \\
&= \mathrm{E}(X) \, \mathrm{E}(Y) - \mathrm{E}(X) \, \mathrm{E}(Y) \\
&= 0 
\end{aligned}
$$

__Variance of the sum of two random variables.__ The variance of the sum of two random
 variables equals the sum of the variances of those random variables, 
 plus two times their covariance:

 $$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2 \, \mathrm{Cov}(X,Y)$$

Using the linearity of expectation we can prove:

$$
\begin{aligned}
\mathrm{Var}(X+Y) &= \mathrm{E}\left[ ((X+Y)-\mathrm{E}(X+Y))^2 \right] \\
&= \mathrm{E}\left[ ([X-\mathrm{E}(X)] + [Y-\mathrm{E}(Y)])^2 \right] \\
&= \mathrm{E}\left[ (X-\mathrm{E}(X))^2 + (Y-\mathrm{E}(Y))^2 + 2 \, (X-\mathrm{E}(X)) (Y-\mathrm{E}(Y)) \right] \\
&= \mathrm{E}\left[ (X-\mathrm{E}(X))^2 \right] + \mathrm{E}\left[ (Y-\mathrm{E}(Y))^2 \right] + \mathrm{E}\left[ 2 \, (X-\mathrm{E}(X)) (Y-\mathrm{E}(Y)) \right] \\
&= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2 \, \mathrm{Cov}(X,Y) 
\end{aligned}
$$

__Variance of the sum of multiple random variables.__ For random variables $X_i$
we have:

$$\text{Var}\left(\sum_{i=1}^{n}X_{i}\right)=\sum_{i=1}^{n}\sum_{j=1}^{n}\text{Cov}\left(X_{i},X_{j}\right)$$


Using the decomposition formula of variance, we have

$${\rm Var} \left( \sum_{i=1}^{n} X_i \right) = E \left( \left[ \sum_{i=1}^{n} X_i \right]^2 \right) - \left[ E\left( \sum_{i=1}^{n} X_i \right) \right]^2$$

Now note that $(\sum_{i=1}^{n} a_i)^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} a_i a_j$, 
which is clear if you think about what you're doing when you calculate 

$$(a_1+...+a_n) \cdot (a_1+...+a_n),$$

by hand. Therefore,

$$E \left( \left[ \sum_{i=1}^{n} X_i \right]^2 \right) = E \left( \sum_{i=1}^{n} \sum_{j=1}^{n} X_i X_j \right) = \sum_{i=1}^{n} \sum_{j=1}^{n} E(X_i X_j)$$

similarly,

$$\left[ E\left( \sum_{i=1}^{n} X_i \right) \right]^2 = \left[ \sum_{i=1}^{n} E(X_i) \right]^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} E(X_i) E(X_j)
$$

So,

$$
\begin{aligned}
{\rm Var} \left( \sum_{i=1}^{n} X_i \right) & = \sum_{i=1}^{n} \sum_{j=1}^{n} \big( E(X_i X_j)-E(X_i) E(X_j) \big) \\ 
& = \sum_{i=1}^{n} \sum_{j=1}^{n} {\rm cov}(X_i, X_j)
\end{aligned}
$$


__Pairwise independence.__ In general:

$$\text{Var}\left(\sum_{i=1}^{n}X_{i}\right)=\sum_{i=1}^{n}\sum_{j=1}^{n}\text{Cov}\left(X_{i},X_{j}\right)$$

If $X_i$ and $X_j$ are independent then $\mathbb{E}X_{i}X_{j}=\mathbb{E}X_{i}\mathbb{E}X_{j}$
and consequently

$$
\text{Cov}\left(X_{i},X_{j}\right):=\mathbb{E}X_{i}X_{j}-\mathbb{E}X_{i}\mathbb{E}X_{j}=0
$$

This leads to:

$$\text{Var}\left(\sum_{i=1}^{n}X_{i}\right)=\sum_{i=1}^{n}\text{Var}X_{i}
$$

_Binomial distribution._ In probability theory and statistics, the binomial distribution with parameters 
$n$ and $p$ is the discrete probability distribution of the number of successes 
in a sequence of n independent experiments, each asking a yes–no question, 
and each with its own Boolean-valued outcome: success (with probability $p$) or 
failure (with probability $q = 1-p$). 

The probability of getting exactly $k$ successes in $n$ independent 
Bernoulli trials is given by the probability mass function:


$$f(k, n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$





























{% endkatexmm %}

