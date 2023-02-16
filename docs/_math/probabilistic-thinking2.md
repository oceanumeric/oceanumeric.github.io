---
title: Develop Some Fluency in Probabilistic Thinking (Part II)
subtitle: The foundation of machine learning and data science is probability theory. In this post, we will develop some fluency in probabilistic thinking with different examples, which prepare data scientists well for the sexist job of the 21st century. 
layout: math_page_template
date: 2023-02-16
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality approximate counting Robert Morris hash table 
published: true
tags: probability algorithm data-science machine-learning bernoulli-trial hypergeometric-distribution binomial-distribution 
---

This the second part of this series of posts as the title has stated. We
will continue to study the material from Cameron Musco's course - [Algorithms for Data Science](https://people.cs.umass.edu/~cmusco/CS514F22/index.html){:target="_blank"} {% cite musco2022 %}. This section assumes readers know what hash table is. If not, just check out 
[this video](https://youtu.be/Nu8YGneFCWE){:target="_blank"} from MIT. 


{% katexmm %}

When we use a random hash function $h(x)$ to map keys into hashing numbers,
we could design a fully random one which has the following property:

$$
\mathbb{P}(h(x) = i) = \frac{1}{n}
$$ 



However, wit the above property, we need to store a table of $x$ values and
their hash values. This would take at least $O(m)$ space and $O(m)$ query time
to look up, making our whole quest for $O(1)$ query time pointless. Therefore
we need to find another property that will avoid collision and $O(1)$ query
time. 


__2-Universial Hash Function__ (low collision probability). A random hash 
function from $h: U \to [n]$ is two universal if :

$$
\mathbb{P}[h(x) = h(y)] \leq \frac{1}{n} \tag{1}
$$

One of the functions that are often used is the following one: 

$$
h(x) = (a x + b \  \mathrm{mod} p) \  \mathrm{mod} \  n 
$$ 



__Pairwise Independent Hash Function.__ A random hash function from
$h: U \to [n]$ is pairwise independence if for all $i, j \in [n]$:

$$
\mathbb{P}[h(x) = i \cap h(y) = j] = \frac{1}{n^2} \tag{2}
$$

We can show that pairwise hash function are 2-universal. Now, for $i \in [n]$,
it has $n-1$ pairs and one pair with itself, therefore:

$$
\begin{aligned}
\mathbb{P}[h(x) = h(y)] & = \sum_{j=1}^n \mathbb{P}[h(x) = i \cap h(y) = j] \\
& = n \cdot \frac{1}{n^2} = \frac{1}{n}
\end{aligned}
$$


## Example 5: Randomized load balancing

We have $n$ requests _randomly_ assigned to $k$ servers. How many requests 
must each server handle? Let's assume $R_i$ as the number of requests 
assigned to server $i$. The expectation tells 

$$
\begin{aligned}
\mathbb{E}[R_i] & = \sum_{j=1}^n \mathbb{E} [ \mathbb{I}_{\text{request j assigned to i}}] \\
& = \sum_{j=1}^n \mathbb{P}(\text{j assigned to i}) \times 1  \\
& = \frac{n}{k} \tag{3}
\end{aligned}
$$

If we provision each server be able to handle twice  the expected load,
what is the probability that a server is overloaded? We can calculate it 
based on Markov's inequality:

$$
\mathbb{P}[R_j  > 2 \mathbb{E}[R_i]] \leq \frac{\mathbb{E}[R_i]}{2\mathbb{E}[R_i]} = \frac{1}{2}
$$


### Chebyshev's inequality 

With a very simple twist, Markov’s inequality can be made
much more powerful. For any random variable $X$ and any value $t > 0$, we have: 

$$
\mathbb{P}(|X| \geq t ) = \mathbb{P}(X^2 \geq t^2 ) \leq \frac{\mathbb{E}[X^2]}{t^2}
$$

Now, by plugging in the random variable $X - \mathbb{E}$, we can hae 
Chebyshev's inequality.


__Chebyshev's inequality.__ For any random variable $X$ and $t > 0$,

$$
 \mathbb{P}(|X - E[X]| \geq t ) \leq \frac{\mathrm{Var}[X]}{t^2} \tag{4}
$$


With Chebyshev's inequality, we can calculate the probability that $X$
falls $s$ standard deviations from its mean as:

$$
\mathbb{P}(|X - E[X]| \geq s \cdot \sqrt{\mathrm{Var}[X]} ) \leq \frac{\mathrm{Var}[X]}{s^2\mathrm{Var}[X]} = \frac{1}{s^2} \tag{5}
$$

Equation (5) gives us the quick estimation on deviation of our estimations. It 
could also shows  that the sample average will always concentrate on 
mean with enough samples $n$ (law of large numbers). 

Consider drawing independent identically distributed (i.i.d) random variables
$X_1, \cdots, X_n$ with mean $\mu$ and variance $\sigma^2$. How well 
does the sample average 

$$ S = \frac{1}{n} \sum_{i=1}^n X_i$$ 

approximate the true mean $\mu$?

First, we need to calculate the sample variance

$$
\mathrm{Var}[S] = \mathrm{Var} \left [ \frac{1}{n} \sum_{i=1}^n X_i \right] = \frac{1}{n^2} \sum_{i=1}^n \mathrm{Var} [X_i] = \frac{1}{n^2} \cdot n \cdot \sigma^2 = \frac{\sigma^2}{n}
$$

By Chebyshev's inequality, for any fixed value $\epsilon > 0$, we have 

$$ 
\mathbb{P}( |S- \mu| \geq \epsilon) \leq \frac{\mathrm{Var}[S]}{\epsilon^2} = \frac{\sigma^2}{n \epsilon^2} \tag{5}
$$

Equation (6) could be used to estimate the size of sample we need for a certain
level of accuracy of the estimation. Instead of using $\epsilon$, we will use 
$\epsilon \cdot \sigma$, then we can have

$$ 
\mathbb{P}( |S- \mu| \geq \epsilon \cdot \sigma) \leq \frac{\mathrm{Var}[S]}{\epsilon^2 \sigma^2} = \frac{1}{n \epsilon^2} \tag{6}
$$

We can do quick estimation on sample size we need to achieve a certain level 
of estimation based on equation (6). Figure 1 shows that the probability 
of our sample mean deviating 10% percent of its variance is less or equal to
$0.1001$ when the sample size is greater than $100$, whereas the probability
is still bigger than 1 if we want our estimation with high accuracy ($\sigma = 0.01$),
which means that one needs a larger sample for that. 


<div class='figure'>
    <img src="/math/images/estimation_sample_size.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Plot of probability against
        sample size based on equation (6).  
    </div>
</div>


Now, back to our balancing load example, we have assumed that each server
could _be able to handle twice the expected load_ and the expected load 
is 

$$\mathbb{E}[R_j] = \frac{n}{k}$$ 

as it has been derived in equation (3) and we want to find the probability that a server is overloaded. We can use 
Chebyshev's inequality to calculate it. However, we need to calculate the 
variance first. Let's denote $R_{i, j}$ as $1$ if $j$ is assigned to 
server $i$ and $0$ otherwise, then 

$$
\begin{aligned}
\mathrm{Var}[R_{i,j}] & = \mathbb{E} \left [ (R_{i, j} - \mathbb{E}[R_{i,j}])^2 \right ] \\
& = \mathbb{P}(R_{i, j}= 1) \cdot (1 - \mathbb{E}[R_{i, j}])^2 + \mathbb{P}(R_{i,j} = 0)(0 - \mathbb{E}[R_{i, j}])^2 \\
& =\frac{1}{k} \cdot\left(1-\frac{1}{k}\right)^2+\left(1-\frac{1}{k}\right) \cdot\left(0-\frac{1}{k}\right)^2 \\
& =\frac{1}{k}-\frac{1}{k^2}  \\
& \leq \frac{1}{k}
\tag{7}
\end{aligned} 
$$

Now, with the _linearity of variance_, we can have 

$$ \mathrm{Var}[R_i] = \sum_{j=1}^n \mathrm{Var}[R_{i, j}]  \leq \frac{n}{k} $$ 


Now, with $\mathbb{E}[R_j] = \frac{n}{k}$ and $ \mathrm{Var}[R_i]  \leq \frac{n}{k}$,
we can apply Chebychev's inequality ($R_i$ is the number of requests sent to
server $i$)

$$
\begin{aligned}
\mathbb{P}(R_i \geq \frac{2n}{k})  & = \mathbb{P}(R_i - \frac{n}{k} \geq \frac{n}{k})  \\
& \leq  \mathbb{P}( |R_i - \frac{n}{k}| \geq \frac{n}{k}) \quad \text{(take absolute value)} \\
&  = \mathbb{P}( |R_i - | \mathbb{E}[R_j] \geq \frac{n}{k}) \\
& \leq \frac{ \mathrm{Var}[R_{i, j}]}{(n/k)^2} \\ 
& \leq \frac{k}{n} \tag{8}
\end{aligned}
$$

Equation (8) shows that overload probability is extremely small when
$k << n!$, which seems counterintuitive as it means bound gets worse as
$k$ grows (more servers do not help). 





{% endkatexmm %}