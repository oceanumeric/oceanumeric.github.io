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
f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right) \tag(2)
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










{% endkatexmm %}