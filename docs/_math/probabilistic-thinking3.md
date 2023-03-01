---
title: Develop Some Fluency in Probabilistic Thinking (Part III)
subtitle: The foundation of machine learning and data science is probability theory. In this post, we will develop some fluency in probabilistic thinking with different examples, which prepare data scientists well for the sexist job of the 21st century. 
layout: math_page_template
date: 2023-02-16
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality approximate counting Robert Morris hash table Bloom filter hashing
published: true
tags: probability algorithm data-science machine-learning bloom-filter gaussian-distribution  
---

In the third post of this series, we will discuss the relationship
between Bernstein inequality and Gaussian distribution. Then, we will
introduce Bloom Filters and its applications. 


## Gaussian distribution

In the last post, we presented the formula of Bernstein inequality as

{% katexmm %}

$$
\mathbb{P} \left [ \left | \sum_{i=1}^N X_i \geq t \right | \right] \leq 2 \mathrm{exp} \left ( - \frac{ t^2 }{2 n \sigma^2 + \frac{2}{3} a \cdot t } \right) \tag{1}
$$


Now, for a Bernoulli variable with $E[X] = 5, \mathrm{Var}[X] = 2.5, N = 10, a = 1$, then we plot the equation (1) with different values of $t$. 


<div class='figure'>
    <img src="/math/images/bernstein_and_gaussian.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Plot of Bernstein inequality and Gaussian distribution with different variances. 
    </div>
</div>

The general form of Gaussian distribution's probability density function is

$$
f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right) \tag{2}
$$

Figure 1 shows the share pf Bernstein inequality is very similar to the 
well-known 'bell shape' of Gaussian distribution. However, Bernstein
inequality gives the probability (value on the curve in Figure 1) directly,
whereas the density function of Gaussian distribution gives the value 
of density (a relative likelihood) and the probability has be calculated 
by taking its integration (area under the curve in Figure 1). 

The Bernstein implies that the distribution of the sum of $n$
bounded independent random variables converges to a Gaussian
(normal) distribution as $n$ goes to infinity. This is the stronger 
version of __Central Limit Theorem__ (CLT). 


Why is the Gaussian distribution is so important in statistics,
science, ML, etc.? Many random variables can be approximated as the sum of a
large number of small and roughly independent random effects.
Thus, their distribution looks Gaussian by CLT. 


We have talked a lot on different inequalities. Here is a [summary](../../pdf/chernoff-notes.pdf){:target="_blank"} from MIT, which you can use it to refresh what we have 
discussed. 


## Bloom filter 

For those who are not very familiar with the concept of Bloom filter,
you could watch this [video](https://youtu.be/V3pzxngeLqw){:target="_blank"}. The idea is very simple: we have a universe database (it could be all possible actions from your target customers) and a small
database on our server, we want to build up a __filter__ to filter out
those 'one-hit-wonders' or 'one-shot-actions' before we doing _expensive_
things (such as query or insert) on our server. Figure 2 gives the illustration. 


<div class='figure'>
    <img src="/math/images/Bloom_filter.png"
         alt="Bloom filter illustration" class="zoom-img"
         style="width:90%"
         />
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Illustration of 
        Bloom filter, which shows how a set of hash function (size = k) and
        a vector of bits (size = m) are able to map the action from a 
        universal database (such as the all urls from the internet) to
        a database on the server. 
    </div>
</div>

As it is shown in Figure 2, we have two layers to do the filter:
i) a set of hash functions, ii) a vector of bits. Let's use an example to show how those two layers working as a filter. 

This is an unrealistically small example; its only purpose is to illustrate the possibility of _false positives_. Choose $m=5$ (number of bits) and
$k = 2$ (number of hash functions):

$$
\begin{aligned}
h_1(x) & = x \ \mathrm{mod} \  5 \\
h_2(x) & = (2x + 3) \ \mathrm{mod} \  5 
\end{aligned}
$$

We first initialize the Bloom filter $B[0:4]$ and then insert $9$ and $11$

|  | $h_1(x)$ | $h_2(x)$ |  |  | B  |  |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Initialize: |  |  | 0 | 0 | 0 | 0 | 0 |  |  |
| Insert 9: | 4 | 1 | 0 | 1 | 0 | 0 | 1 |  |  |
| Insert 11: | 1 | 0 | 1 | 1 | 0 | 0 | 1 |  |  |

Now let us attempt some membership queries:

| | $h_1(x)$ | $h_2(x)$ | Answer |
| :--- | :---: | :---: | --- :|
|Query 15: | 0 | 3 | No, not in B (correct answer) |
|Query 16: | 1 | 0 | Yes, in B (wrong answer: false positive) |

The above exam shows that Bloom filter guarantees that it gives the 
correct answer if the element was not in the database (then new element
could be collected for sure). However it gives the wrong answer if the 
element was not there sometimes (false positive). 

Intuitively, this means the Bloom filter will record any new element 
in its own vector as it is shown in Figure 2 but will only pass the 
element into the next stage for caching or retrieving if it sees the 
element again (which it could make mistakes because of the false positive issue). 

Now, we will analyze how does the false positive rate $\delta$ depend on
$m, k$ and the number of items inserted. 

First, we want to find out: what is the probability that after inserting $n$ elements, the $i$th bit of the array $B$ is still $0$? (meaning $n \times k$ total hashes must not hit bit $i$)

$$
\begin{aligned}
\mathbb{P}[B[i] = 0] & = \mathrm{Pr} \left (h_1(x_1) \neq i \cap h_2(x_1) \neq i \cap \cdots \cap h_k(x_1) \neq i \right ) \\
& \times \mathrm{Pr} \left (h_1(x_2) \neq i \cap h_2(x_2) \neq i \cap \cdots \cap h_k(x_2) \neq i \right ) \times \\ 
& \cdots \times \\
& \mathrm{Pr} \left (h_1(x_n) \neq i \cap h_2(x_n) \neq i \cap \cdots \cap h_k(x_n) \neq i \right ) \\
& = \left( 1 - \frac{1}{m} \right )^{kn} \\
& = \left (  1 + \frac{-n/m}{n} \right)^{kn} \\
& \approx e^{-\frac{kn}{m}}
\end{aligned}
$$

Now, we will calculate the probability that querying a new item $w$ gives a
false positive. 

$$
\begin{aligned}
\mathbb{P}[B[j] = 1] & = \mathrm{Pr} \left (h_1(w) = j \cap h_2(w) = j \cap \cdots \cap h_k(w) = j \right ) \\ 
& = \left( 1 - \mathbb{P}[B[j] = 0] \right )^k \\ 
& = \left( 1 - e^{-\frac{kn}{m}} \right)^k 
\end{aligned}
$$

Therefore, as it is shown in Figure 2, with $m$ bits of storage, $k$ hash
functions, and $n$ elements inserted, the false positive rate is

$$
\delta = \left( 1 - e^{-\frac{kn}{m}} \right)^k  \tag{3}
$$

How could set the value of $k$ to o minimize the
false positive rate given a fixed amount of space $m$ and $n$? Now, 
let $f =$ equation (3), then take the log:

$$
g = \ln(f) = k \cdot \ln \left ( 1 - e^{-\frac{kn}{m}}  \right)
$$

We take the derivative, 

$$
\frac{d g}{d k}=\ln \left(1-e^{-\frac{k n}{m}}\right)+\frac{k n}{m} \cdot \frac{e^{-\frac{k n}{m}}}{1-e^{-\frac{k n}{m}}}
$$

We find the optimal $k$, or right number of hash functions to use, when the derivative is $0$. This occurs when

$$
k = (\ln 2)  \cdot \frac{m}{n} \tag{4}
$$

Now substitute the value in equation (4) into the equation (3), the optimal
false positive rate

$$
\tilde{\delta} = \left( \frac{1}{2} \right)^{\ln2 \quad \frac{m}{n}} \approx (0.6185)^{\frac{m}{n}}
$$





{% endkatexmm %}