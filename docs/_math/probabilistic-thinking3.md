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














{% endkatexmm %}