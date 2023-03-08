---
title: Approximating Distinct Element in a Stream
subtitle: This post explains a probabilistic counting algorithm with which one can estimate the number of distinct elements in a large collection of data in a single pass. 
layout: math_page_template
date: 2023-03-08
keywords: probabilistic-thinking approximating distinct elements
published: true
tags: probability algorithm data-science machine-learning bloom-filter online-learning
---


This post is based on notes by [Alex Tsun](https://www.alextsun.com/index.html){:target="_blank"} and paper by Flajolet and Martin (1985). It is pretty amazing the the algorithm discovered
in the 1980s when the memory was very expensive has a huge number
of applications in networking, database, and computer system  now - big data era {% cite flajolet1985probabilistic %}. 


## Motivation problem


{% katexmm %}

For one of the most popular videos on Youtube, let's say there
are $N = 2$ billion views, with $n = 900$ million of them being
distinct views. How much space is required to accurately track 
this number? Well, let's assume a user ID is an 8-byte integer. Then,
we need $900,000,000 \times 8$ bytes total if we use a Set to track
the user IDs, which requires $7.2$ gigabytes of memory for ONE video.


## Intuitions and examples

When we count distinct values $n$, we notice that it has the following
property:

- $n >= 0$ for sure
- $n$ is monotonic increasing 

For instance, we cannot have $-2$ distinct values (distinct users). Furthermore, the distinct values always increase or remain constant, such as we have $7$ distinct users at the current moment and this value will either increase or remain constant during the streaming process. 

Instead of having a set of distinct values and querying the new 
value is unique or not, we could rely on hashing and probability
to count the number of distinct values with $O(1)$ space. 


<div class='figure'>
    <img src="/math/images/distinct-values.png"
         alt="distinct values"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Illustration of
        using hash function to count the number of distinct values.
    </div>
</div>

The idea of approximating distinct elements in a stream is to use the
minimum value generated randomly from the hash function as it is 
illustrated in the Figure 1. 


## Probabilistic reasoning

If $U_1, \cdots, U_m \sim \mathrm{Unif}(0, 1)$ (continuous) are
iid, and $X = \mathrm{min}\{U_1, \cdots, U_m \}$ is their minimum where
$m$ is the number of distinct values, then 

$$
\mathbb{E}[X] = \frac{1}{m+1} \tag{1}
$$

To prove the above statement, we start working with probabilities
first (e.g., the CDF $F_X(x) = \mathbb{P}(X \leq x)$) and take the 
derivative to find the PDF (this is a common strategy for dealing
with continuous random variables). 

$$
\begin{aligned}
\mathbb{P}(X > x) & = \mathbb{P}(\mathrm{min}\{U_1, \cdots, U_m\} > x) \\
& = \mathbb{P}(U_1 > x, U_2 >x, \cdots, U_m >x )\\
& = \prod_{i = 1}^m \mathbb{P}(U_i > x) \\
& = \prod_{i=1}^m (1-x) \\
& = (1-x)^m \tag{2}
\end{aligned}
$$

For the second equation, we use the fact that the minimum of numbers is greater than a value if and only if all of them are. 

Now, we have that 

$$
\mathrm{F}_X(x) = \mathbb{P}(X \leq x) = 1 - \mathbb{P}(X > x) = 1 - (1-x)^m \tag{3}
$$

Now, take the derivative of Equation (3), we can have 

$$
f_X(x) = m(1-x)^{m-1} \tag{4}
$$

Now, we can calculate the expectation of random variable $X$

$$
\begin{aligned}
\mathbb{E}[X] & = \int_0^1 x f_X(x) dx \\
            & = \int_0^1 x  m(1-x)^{m-1} dx \\
            & = m \int_0^1 x  (1-x)^{m-1} dx \\
            & = m \int_1^0 (u-1)  u^{m-1} du  \quad (u = 1-x; du = -dx) \\
            & =  m \int_1^0 (u^m - u^{m-1}) du  \\
            & = m \left [ \frac{1}{m+1}u^{m+1} - \frac{1}{m} u^m + C \right]_1^0 \\ 
            & = m \left [ \frac{1}{m} - \frac{1}{m+1} \right] \\
            & = \frac{1}{m+1} \tag{5}
\end{aligned}
$$

The Equation (5) shows that we could calculate the value of $m$ as 
follows:

$$
m := \frac{1}{\mathbb{E}[X]} - 1 
$$

For instance, we have $\mathbb{E}[X] = 0.32$, the number of distinct
values can be calculated as

$$
m := \frac{1}{0.32} -1 \approx 3.125 -1 = 2.125 
$$








{% endkatexmm %}

