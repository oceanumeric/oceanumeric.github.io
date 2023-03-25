---
title: The Johnson Lindenstrauss Lemma
subtitle: In the era of AI, the Johnson Lindenstrauss lemma provides the mathematical foundation for many applications of machine learning and deep learning, such as ChatGPT. 
layout: math_page_template
date: 2023-03-25
keywords: probabilistic-thinking linear algebra Johnson Lindenstrauss lemma
published: true
tags: probability algorithm data-science machine-learning
---

Very high-dimensional vectors are ubiquitous in science, engineering, and machine learning.
They give a simple way of representing data either in a vector or a matrix, which then could be analyzed with all those linear algebra and probability tools. 

For genetic data sets, images, time series (e.g. audio), texts, the number of features is normally large, which makes them become high-dimensional data easily. 



What do we want to do with such high dimensional vectors? Cluster them, use them
in regression analysis, feed them into machine learning algorithms. As an even more basic
goal, all of these tasks require being able to determine if one vector is _similar_ to another. However, human beings are not very good at
analyzing high dimensional features as the complexity and computing cost of solving any problems of big and high dimensional datasets grow
exponentially.

Therefore, we need to transfer high dimensional vectors into low dimension and ask: how much of the feature and structure of high dimensional data will be preserved if we map it into the lower dimensional ones? 

The Johnson Lindenstrauss lemma gives the answer. This blog is based on the course notes from [Anupam Gupta](http://www.cs.cmu.edu/~anupamg/){:target="_blank"}, [MIT](http://courses.csail.mit.edu/6.854/16/Notes/n6-dim_reduction.html){:target="_blank"}, and the book by Blum et al {% cite blum2020foundations %}. 

## Markov's inequality and the Chernoff bound

{% katexmm %}

Before we give the Johnson Lindenstrauss lemma and prove it, let's review some key inequalities in probability. 

(Markov's inequality). Let $X \geq 0$ be a nonnegative random variable, then for any $a > 0$, 

$$
\mathbb{P}(X \geq a) \leq \frac{\mathbb{E}[X]}{a}
$$

Based on Markov's inequality, we can have

$$
\begin{aligned}
\mathbb{P}(X \geq a) & = \mathbb{P}(t x \geq ta)  \\ 
                    & = \mathbb{P}(e^{tx} \geq e^{ta}) \\
                    & \leq e^{-ta} \mathbb{E}[e^{tx}],
\end{aligned}
$$

This is the famous _Chernoff bound_. 

Now, let's see an example of this bound in action. Let $X \sim \mathcal{N}(\mu, \sigma^2)$, then $e^X$ follows the log-normal distribution, we can compute 

$$
\mathbb{E}[e^{t(X-\mu)}] = e^{\frac{t^2\sigma^2}{2}}
$$

Plugging the above equation into the Chernoff bound gives 

$$
\mathbb{P}(X- \mu \geq a) \leq e^{-ta} e^{\frac{t^2\sigma^2}{2}} = \exp \left ( \frac{t^2 \sigma^2}{2} - ta  \right)
$$

For the equation 

$$
\frac{t^2 \sigma^2}{2} - ta,
$$

We know that it has the minimum value at  $ t = a/\sigma^2$, which means we can have 

$$
\begin{aligned}
\mathbb{P}(X- \mu \geq a) \leq e^{-ta} e^{\frac{t^2\sigma^2}{2}} & = \exp \left ( \frac{t^2 \sigma^2}{2} - ta  \right)\\
&  = \exp \left( \frac{a^2}{2 \sigma^2} - \frac{a^2}{\sigma^2} \right) \\
& = \exp \left( - \frac{a^2}{2\sigma^2} \right)
\end{aligned}
$$

We get the same thing if we apply the argument to $-X$, so adding the two cases together gives 

$$
\mathbb{P}(|X-\mu| \geq a) \leq 2 \exp \left( - \frac{a^2}{2\sigma^2} \right)
$$




## The Johnson Lindenstrauss lemma



Given any set of points $X = \{ x_1, x_2, \cdots, x_n \}$ in $\mathbb{R}^d$, and any $\epsilon \in (0, 1/2]$, there exists a linear map  $\Pi: \mathbb{R}^d \to \mathbb{R}^m$ with
$m = O( \frac{\log n}{\epsilon^{2}})$ such that 

$$
1- \epsilon \leq \frac{||\Pi(x_i) - \Pi(x_j) ||_2^2}{||x_i - x_j||_2^2} \leq 1+ \epsilon \tag{1}
$$

for any $i$ and $j$. 

- Since it is a linear map, which means $\Pi$ is just a function of a matrix, let's say this matrix is $A$
- Intuitively, the above statement tells us that there is a mapping of $d$-dimensional vectors to $m$-dimensional vectors, with $m=O( \frac{\log n}{\epsilon^{2}})$ that keep the $\ell_2$-norm distances be (approximately) the same.
- Note that this mapping is linear, which is a very useful property in many of the applications.
- Observe that $m$ _does not_ depend on $d$
- Additionally, even though, in principle, in the statement above the matrix $A$ could depend on the vectors $x_1, \cdots, x_n$, we will construct it in a way that is completely _oblivious_ to what these vectors are. 
- However, as one could show that there is no single $A$ that works 
for all collections of vectors, we will make the construction of $A$ probabilistic and prove that for any fixed set of vectors it works with high probability. 
- specifically, we will make each entry of $A$ be sampled independently
from a Gaussian distribution $\mathcal{N}(0, \frac{1}{m})$, where the density of a Gaussian random variable is given by


$$
\mathcal{N}(\mu, \sigma^2, x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left ( - \frac{(x-\mu)^2}{2 \sigma^2} \right)
$$




Now, let $A$ be a $m \times d$ matrix, such that every entry 
of $A$ is filled with an i.i.d draw from a standard normal
$a_{i, j} \sim N(0, 1/m)$ distribution (a.k.a the "Gaussian" distribution). For $x \in \mathbb{R}^d$, define 

$$
\Pi(x) = A x, \tag{2}
$$

<div class='figure'>
    <img src="/math/images/johnson-lindenstrauss-lemma.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Illustration of random project process with matrix A. 
    </div>
</div>


Now, we will establish a certain norm preservation property of $A$.
Specifically, it suffices to prove that, for any fixed vector $x \in \mathbb{R}^d$, we have that

$$
\mathbb{P} \bigg [ (1- \epsilon) ||x||_2^2 \leq ||A x||_2^2 \leq (1+\epsilon) ||x||_2^2 \bigg ] \geq 1 - \frac{1}{n^3} \tag{3}
$$

To prove equation (3), let us fix $x \in \mathbb{R}^d$, and consider
the distribution of the $i$-th coordinate $Y_i$ of the mapping $Ax$ 
of $x$ 

$$
Y_i := (Ax)_i = \sum_{j=1}^d A_{i, j} x_j \tag{4}
$$

Each random variable $Y_i$ is a sum of Gaussian distributions. Specifically, we have that 

$$
Y_i \sim \sum_{j=1}^d \mathcal{N} \left (0, \frac{1}{m} \right ) x_j \sim  \sum_{j=1}^d \mathcal{N} \left(0, \frac{x_j^2}{k} \right ) \sim \mathcal{N} \left (0, \frac{||x||_2^2}{k} \right ), \tag{5}
$$

where we used the crucial property of Gaussian distributions that they are closed (linear) under taking a sum. More precisely,
for any two Independent Gaussian random variables $Z_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Z_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$, we have that 

$$
Z_1 + Z_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2).
$$

Consequently, we can conclude that 

$$
\mathbb{E} \big [ ||Ax||_2^2 \big] = \mathbb{E} \left [ \sum_{i=1}^m Y_i^2  \right ]= \sum_{i}^m \mathbb{E}[Y_i^2] = m \cdot \mathrm{Var}(Y_i) = m \cdot \frac{||x||_2^2}{m} = ||x||_2^2 . \tag{6}
$$

This means $\ell_2$ norm is preserved, but how about the concentration around this expectation? We need a tail bound. 

Recall that $||Ax||_2^2$ is distributed as Gaussian variable. We know that Gaussian variables are very concentrated around expectation - the tail vanishes at an exponential rate. For instance, $\mathcal{N}(\mu, \sigma^2)$ is within $2 \sigma$ of the expectation with probability at least $0.95$. 

This strong concentration is quantified via _Chernoff bounds_ (for $\chi$-squared distributions). 









{% endkatexmm %}