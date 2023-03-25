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

The Johnson Lindenstrauss lemma gives the answer. This blog is based on the course notes from [Anupam Gupta](http://www.cs.cmu.edu/~anupamg/){:target="_blank"} and the book by {% cite blum2020foundations %}. 

## The Johnson Lindenstrauss lemma

{% katexmm %}

Given any set of points $X = \{ x_1, x_2, \cdots, x_n \}$ in $\mathbb{R}^D$, there exists a map $\Pi: \mathbb{R}^D \to \mathbb{R}^m$ with
$m = O( \frac{\log n}{\epsilon^{2}})$ such that 

$$
1- \epsilon \leq \frac{||\Pi(x_i) - \Pi(x_j) ||_2^2}{||x_i - x_j||_2^2} \leq 1+ \epsilon \tag{1}
$$

Note that the target dimension $m$ is independent of the original 
dimension $D$, and depends only on the number of points $n$ and the accuracy parameter $\epsilon$. 

Now, we will try yo construct the matrix for map $\Pi$. The JL lemma is really amazing, but the construction of the map
is perhaps even more surprising: it is a super-simple random construction. 

Now, let $M$ be a $m \times D$ matrix, such that every entry 
of $M$ is filled with an i.i.d draw from a standard normal
$M_{i, j} \sim N(0, 1)$ distribution (a.k.a the "Gaussian" distribution). For $x \in \mathbb{R}^D$, define 

$$
\Pi(x) = \frac{1}{\sqrt{k}} M x, \tag{2}
$$

Now, we will show that if $m = O(\frac{\log (1/\delta)}{\epsilon^2})$ , then for any vector $x$,

$$
(1- \epsilon) ||x||_2^2 \leq ||\Pi x||_2^2 \leq (1+\epsilon) ||x||_2^2 \tag{3}
$$

with probability $(1-\delta)$. Note that, while we stated with the squared Euclidean norm, equation (3) immediately implies that the $L1$ normal also holds. 

A few comments about this construction:

- 













{% endkatexmm %}