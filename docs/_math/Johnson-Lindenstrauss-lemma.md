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

The Johnson Lindenstrauss lemma gives the answer. This blog is based on the course notes from [Anupam Gupta](http://www.cs.cmu.edu/~anupamg/){:target="_blank"}, [Daniel Raban](https://pillowmath.github.io/){:target="_blank"}, [MIT](http://courses.csail.mit.edu/6.854/16/Notes/n6-dim_reduction.html){:target="_blank"}, and the book by Blum et al {% cite blum2020foundations %}. 

- [Markov's inequality and the Chernoff bound](#markovs-inequality-and-the-chernoff-bound)
- [The Johnson Lindenstrauss lemma](#the-johnson-lindenstrauss-lemma)
- [Comments about the Johnson Lindenstrauss lemma](#comments-about-the-johnson-lindenstrauss-lemma)

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
\mathbb{P}(|X-\mu| \geq a) \leq 2 \exp \left( - \frac{a^2}{2\sigma^2} \right) \tag{1}
$$

Now, we construct a random variable $\bar{X}$ such that it is an average of i.i.d. random variables $X_1, \cdots, X_n$, then we have

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i.
$$

Then, $\bar{X}$ follows the normal distribution with mean $\mu$ and variance $\frac{\sigma^2}{n}$. Then we apply the Chernoff bound to $\bar{X}$, we have

$$
\mathbb{P}(|\bar{X}-\mu| \geq a) \leq 2 \exp \left( - \frac{n a^2}{2\sigma^2 } \right)
$$

This type of inequality is called the _concentration inequality_.
It tells us that the probability of a random variable being far away from its mean concentrates around the mean at an exponential rate.

Using the above inequality, we can prove the Johnson Lindenstrauss lemma.

We know that a random variable $X$ follows the Chi-square distribution with $m$ degrees of freedom if $X = \sum_{i=1}^m Z_i^2$, where $Z_i \sim \mathcal{N}(0, 1)$ are i.i.d. random variables. 

We could calculate the first and second moments of $X$ as

$$
\begin{aligned}
\mathbb{E}[X] & = \sum_{i=1}^m \mathbb{E}[Z_i^2] \\
                & = m \\
\mathbb{V}[X] & = \sum_{i=1}^m \mathbb{V}[Z_i^2] \\
                & = 2m \\
\end{aligned}
$$

This means for the average of $X$, we have

$$
\begin{aligned}
\mathbb{E}[\bar{X}] & = \frac{1}{m} \sum_{i=1}^m \mathbb{E}[Z_i^2] \\
                    & = \frac{1}{m} m \\
                    & = 1 \\
\mathbb{V}[\bar{X}] & = \frac{1}{m} \sum_{i=1}^m \mathbb{V}[Z_i^2] \\
                    & = \frac{1}{m} 2m \\
                    & = \frac{2}{m} \\
\end{aligned}
$$

This gives us the following inequality if we substitute the above values ($\mu=1, \mathbb{V}=2/m$) into the Chernoff bound in the equation (1). 

$$
\mathbb{P}(|\bar{X}-1| \geq a) \leq 2 \exp \left( - \frac{m a^2}{8} \right)
$$

Or with the variable $Z$ involved, we have

$$
\mathbb{P} \left(\left | \frac{1}{m} \sum_{i=1}^m Z_i^2 - 1 \right | \geq a \right ) \leq 2 \exp \left( - \frac{m a^2}{8} \right), \tag{2}
$$

which is equivalent to

$$
\mathbb{P} \left( \left | \sum_{i=1}^m Z_i^2 - m \right | \geq ma  \right ) \leq 2 \exp \left( - \frac{m a^2}{8} \right). \tag{3}
$$

Now, we are ready to prove the Johnson Lindenstrauss lemma.


## The Johnson Lindenstrauss lemma


Given any set of points $X = \{ x_1, x_2, \cdots, x_n \}$ in $\mathbb{R}^d$, and any $\epsilon \in (0, 1/2]$, there exists a linear map  $\Pi: \mathbb{R}^d \to \mathbb{R}^m$ with
$m = O( \frac{\log n}{\epsilon^{2}})$ such that 

$$
1- \epsilon \leq \frac{||\Pi(x_i) - \Pi(x_j) ||_2^2}{||x_i - x_j||_2^2} \leq 1+ \epsilon \tag{4}
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
\Pi(x) = A x, \tag{5}
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
\mathbb{P} \bigg [ (1- \epsilon) ||x||_2^2 \leq ||A x||_2^2 \leq (1+\epsilon) ||x||_2^2 \bigg ] \geq 1 - \frac{1}{n^3} \tag{6}
$$

To prove equation (6), let us fix $x \in \mathbb{R}^d$, and consider
the distribution of the $i$-th coordinate $Y_i$ of the mapping $Ax$ 
of $x$ 

$$
\begin{aligned}
& Y_i := (Ax)_i  = \sum_{j=1}^d A_{i, j} x_j  \\
& Ax  = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_m \end{bmatrix}
\tag{7}
\end{aligned}
$$

Each random variable $Y_i$ is a sum of Gaussian distributions. Specifically, we have that 

$$
Y_i \sim \sum_{j=1}^d \mathcal{N} \left (0, \frac{1}{m} \right ) x_j \sim  \sum_{j=1}^d \mathcal{N} \left(0, \frac{x_j^2}{k} \right ) \sim \mathcal{N} \left (0, \frac{||x||_2^2}{k} \right ), \tag{8}
$$

where we used the crucial property of Gaussian distributions that they are closed (linear) under taking a sum. More precisely,
for any two Independent Gaussian random variables $Z_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Z_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$, we have that 

$$
Z_1 + Z_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2).
$$

Consequently, we can conclude that 

$$
\mathbb{E} \big [ ||Ax||_2^2 \big] = \mathbb{E} \left [ \sum_{i=1}^m Y_i^2  \right ]= \sum_{i}^m \mathbb{E}[Y_i^2] = m \cdot \mathrm{Var}(Y_i) = m \cdot \frac{||x||_2^2}{m} = ||x||_2^2 . \tag{9}
$$

This means $\ell_2$ norm is preserved, but how about the concentration around this expectation? We need a tail bound. 

Recall that $||Ax||_2^2$ is distributed as Gaussian variable. We know that Gaussian variables are very concentrated around expectation - the tail vanishes at an exponential rate. For instance, $\mathcal{N}(\mu, \sigma^2)$ is within $2 \sigma$ of the expectation with probability at least $0.95$. 

This strong concentration is quantified via _Chernoff bounds_ (for $\chi$-squared distributions). 

In equation (9), we have shown 

$$
\mathbb{E} \big [ ||Ax||_2^2 \big] = \mathbb{E} \left [ \sum_{i=1}^m Y_i^2  \right ]= ||x||_2^2 \tag{10}
$$

In equation (10), we have shown

$$
Y_i \sim \mathcal{N} \left (0, \frac{||x||_2^2}{k} \right ), \tag{11}
$$

Now, we standardize the random variable $Y_i$ as follows:

$$
\frac{\sqrt{m}}{||x||_2} Y_i \sim \mathcal{N} \left (0, 1 \right ). \tag{12}
$$

This means $\sum_{i}^m Y_i^2$ follows the chi-squared distribution with $m$ degrees of freedom.

Now, we can apply the Chernoff bound to the random variable $\sum_{i}^m Y_i^2$ to obtain the following bound:

$$
\begin{aligned}
\mathbb{P} \left [ \bigg | ||Ax||_2^2 - \mathbb{E}[ ||Ax||_2^2] \bigg | \geq \epsilon ||x||_2^2 \right] & = \mathbb{P} \left [ \bigg | ||Ax||_2^2 - ||x||_2^2 \bigg  | \geq \epsilon ||x||_2^2 \right] \\
& = \mathbb{P} \left [ \bigg | \sum_{i}^m Y_i^2 - ||x||_2^2 \bigg | \geq \epsilon ||x||_2^2 \right] \\
& = \mathbb{P} \left [ \left | \sum_{i}^m  \left( \frac{\sqrt{m}}{||x||_2} Y_i  \right)^2 - m \right | \geq \epsilon m \right] \\ 
& \leq 2 \exp \left ( - \frac{\epsilon^2 m}{8} \right ) \tag{13}
\end{aligned}
$$

Now, if we set the value of $m$ as follows:

$$
m \geq  16 \cdot \epsilon^{-2} \log (\frac{n}{\sqrt{\epsilon}}) \tag{14}
$$

Then we could have 

$$
\begin{aligned}
\mathbb{P} \left [ \bigg | ||Ax||_2^2 - ||x||_2^2 \bigg  | \geq \epsilon ||x||_2^2 \right] & \leq 2 \exp \left ( - 2 \log (n/\sqrt{\epsilon}) \right )  \\
& =  2 \exp(\log n/\sqrt{\epsilon})^{-2} \\
& = \frac{2\epsilon}{n^2}
\end{aligned}
$$

Now, if we treat all the pairs of vectors $x$ in $\mathbb{R}^d$ as a random sample, then we can apply the union bound to obtain the following bound:

$$
\begin{aligned}
& \mathbb{P} \left [ \bigg | ||A(x_i) - A(x_i)||_2^2 - ||x_i- x_j||_2^2 \bigg  | \geq \epsilon ||x_i - x_j||_2^2 \right] \\ 
& \quad \quad \quad \quad  = \mathbb{P} \left [ \left | \frac{||A(x_i) - A(x_i)||_2^2 }{||x_i- x_j||_2^2} - 1 \right | \geq \epsilon \right] \\ 
& \quad \quad  \quad \quad \leq \sum_{i \neq j} \mathbb{P} \left [ \bigg | ||A(x_i) - A(x_i)||_2^2 - ||x_i- x_j||_2^2 \bigg  | \geq \epsilon ||x_i - x_j||_2^2 \right] \\
& \quad \quad  \quad \quad \leq { n \choose 2} \cdot \frac{2\epsilon}{n^2}  \\
& \quad \quad  \quad \quad \leq \epsilon
\end{aligned}
$$

This concludes the proof $\square$. 


## Comments about the Johnson-Lindenstrauss Lemma


- The above proof shows not only the existence of a good map, we also get that a random
map as above works with constant probability! In other words, a Monte-Carlo randomized
algorithm for dimension reduction. (Since we can efficiently check that the distances are
preserved to within the prescribed bounds, we can convert this into a Las Vegas algorithm.)

- The algorithm (at least the Monte Carlo version) does not even look at the set of points $X$: it
works for any set $X$ with high probability. Hence, we can pick this map $A$ before the points in $X$ arrive. 

- Two issues: the matrix A
 we sample above has real-valued entries, and it is very dense, i.e., each entry is non-zero (with probability 1). This is not very practical.

 - To deal with the first issue, we can replace the matrix $A$ with a random matrix $A$ with i.i.d. entries, and then normalize the columns of $A$ to have unit norm. This is called the _Gaussian Random Projection_.

 - the second method is to use a _sparse_ random matrix $A$. This is called the _Sparse Random Projection_. The paper by Kane and Nelson (2014) shows there exists a way of sampling A
 to ensure that there is only at most $O(\frac{\log n}{\epsilon})$
 non-zero entries per column. (The distribution here cannot correspond to simple entry-wise independent sampling anymore.) {% cite kane2014sparser %}. 
 
 - The paper by Achlioptas et al. (2003) shows that sampling each entry of $A$ independently from a Bernoulli distribution with parameter $p$ works well. For instance, $+1$ with probability $1/6$, $-1$ with probability $1/6$, and $0$ with probability $2/3$. This is called the _Bernoulli Random Projection_ {% cite achlioptas2003database %}. 

 - Natural question: Can we have such a strong dimensionality reduction phenomena also for other distances, e.g., $\ell_1$ or
 $\ell_{\inf}$ norms?

 - Unfortunately, no. It turns out that $\ell_2$-norm is very special. 


{% endkatexmm %}