---
title: Dive into Latent Dirichlet Allocation Model
subtitle: Not many models balance complexity and effectiveness as well as LDA. I like this model so much as it is perhaps the best model to start with when you want to learn about machine learning and deep learning models. Why? I will explain in this post.
layout: blog_default
date: 2023-04-19
keywords: Latent Dirichlet Allocation, LDA, Topic Modeling, Machine Learning, Deep Learning
published: true
tags: machine-learning deep-learning topic-modeling 
---

What we talk about when we talk about machine learning and deep learning? It is hard to say as there are so many things going on right now. They say one should follow a saint in turbulent times. Who should we follow and where should we start then? I think we should start with the basics and follow our intuition and logics. 

So, let's quote 'the Godfater' of deep learning, Geoffrey Hinton, who said: _"The deep learning revolution has transformed the field of machine learning over the last decade. It was inspired by attempts to mimic the way the brain learns but it is grounded in basic principles of statistics, information theory, decision theory and optimization."_. 

To illustrate the above quote, I will use a simple model called Latent Dirichlet Allocation (LDA) to explain the basic principles of machine learning and deep learning. I think after understanding LDA, one could easily understand the basic principles of machine learning and deep learning. 

This post is based on the paper by the author of LDA, David Blei, titled _Probabilistic Topic Models_ {% cite blei2012probabilistic %}. I will recommend you to read the paper after reading this post. Here is the roadmap of this post:

- [The big picture](#the-big-picture)
- [The probability distributions](#the-probability-distributions)
- [The estimation of LDA](#the-estimation-of-lda)
- [A simple example](#a-simple-example)

## The big picture

Please look at the following figure. Notice that words are <mark style="background-color:#FA3C92">highlighted</mark> <mark style="background-color:#7FFDFD">in different</mark> <mark>colors</mark>. This is because we want to show that words are grouped into different topics.


<div class='figure'>
    <img src="/images/blog/lda-illustration-blei.png"
         alt="fp7 totalcost hist1" class="zoom-img"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of LDA from the paper by David Blei (2012). You can click on the image to zoom in.
    </div>
</div>

In the above article titled _Seeking Life's Bare (Genetic) Necessities_, Four topics are highlighted: topic 1 is about <mark>genetics</mark>, topic 2 is about  <mark style="background-color:#FA3C92">evolutionary biology</mark>, topic 3 is about <mark style="background-color:#8DCECE">data analysis</mark> and topic 4 is about <mark style="background-color:#95CA56">others</mark>.



| genetics | evolution | data analysis | others |
|:----------|:-----------|:---------------|:--------|
| gene 0.04 | life 0.02 | data 0.02 | brain 0.04 |
|dna 0.02 | evolve 0.01 | neuron 0.02 | number 0.02 | 


This is how LDA works. It groups words into topics and it assumes that each document is a mixture of topics. This is the big picture of LDA, which really aligns with our common sense and intuition. When we read an article or a newspaper, we always try to identify the thesis (or the theme) and the topics of the article. If the theme is to deliver a key message, then the topics are the supporting arguments or supporting evidences.

## The probability distributions

{% katexmm %}

After understanding the big picture of LDA, let's dive into the details. The first thing we need to understand is the probability distributions. There are three probability distributions in LDA: the document-topic distribution, the topic-word distribution and the word distribution. 

If you look at the Figure 1, you will see that we have present:

- several documents that are usually called _corpus_ in machine learning and deep learning
- one document
- several topics that are usually called _latent variables_ in machine learning and deep learning

Our goal is to extract the topics from the _corpus_ and assign each document to one or more topics. But how could we find the topics? Or another question we need to think: how many documents do we need to find a set of topics? Do you think an algorithm could find the topics by reading one document? Why? Can human find the topics by reading one document? Why? 

For a human being, it is easy to find the topics by reading one document if she or he was educated and familiar with the topic. But for an algorithm, it is not easy to find the topics by reading one document. Why? Because the algorithm does not have any prior knowledge about the topics. So, we need to give the algorithm many documents to find the topics, which means learning from the data. 

To let algorithm learn from the data, we have to design a model and algorithm to let it learn. To do this, we need to think __how a document is generated__. 

<div class='figure'>
    <img src="/images/blog/lda-illustration1.png"
         alt="fp7 totalcost hist1" class="zoom-img"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The illustration of LDA in terms of generating a document. You can click on the image to zoom in.
    </div>
</div>

In Figure 2, assume we have $M$ documents making up the _corpus_. Each document is a mixture of $K$ topics. Each topic is a mixture of $V$ words. Notice that a document does not have to use every word in the dictionary.

Now, we define a document $w$ as a finite sequence of of $N$ words:

$$
w = \{w_1, w_2, \cdots, w_N\}
$$

and denote a corpus as $D$ of $M$ documents:

$$
D = \{w_1, w_2, \cdots, w_M\}
$$

We assume there are $k$ topics in the corpus and for a document $w_i$ in the corpus with length $N_i$, we have the following generating process:

0. for the length of document, we assume $N_i \sim \text{Poisson}(\lambda)$

1. for a vector $\theta_i \in \mathbb{R}^{k}$, which follows a $k$-dimensional Dirichlet distribution with parameter $\alpha$: $\theta_i \sim \text{Dirichlet}(\alpha)$

2. for each word $w_{ij}$ in the document $w_i$ with index $j$:

    1. sample a topic $z_{ij}$ from a $k$-dimensional multinomial distribution with parameter $\theta_i$: $z_{ij} \sim \text{Multinomial}(\theta_i)$

    2. sample a word $w_{ij}$ from a $v$-dimensional multinomial distribution with parameter $\beta_{z_{ij}}$: $w_{ij} \sim \text{Multinomial}(\beta_{z_{ij}})$

where $\alpha$ and $\beta$ are hyperparameters, where $\alpha \in \mathbb{R}^{k}$ and $\beta \in \mathbb{R}^{k \times V}$.

If we put everything into a matrix, we have:

$$
\underbrace{
\begin{bmatrix}
\theta_{11}  & \cdots & \theta_{1k} \\
& &   \\
&   M \times K  & \\
& &   \\
\theta_{m1}  & \cdots & \theta_{mk} 
\end{bmatrix}}_{\text{Topic assignment} \ \theta \sim \text{Dir}(\alpha) } \ \  \underbrace{\begin{bmatrix}
 & & & & \\
& & & & \\
& & K \times V & & \\
& & & & \\
& & & &
\end{bmatrix}}_{\beta} \approx \underbrace{\begin{bmatrix}
 & & & & \\
& & & & \\
& & N \times V & & \\
& & & & \\
& & & &
\end{bmatrix}}_{\text{Corpus}} \tag{1}
$$

The original paper by Blei et al. did not use the above matrix representation. But they did give the following graphical representation, which is very helpful to understand the generating process of LDA. 

<div class='figure'>
    <img src="/images/blog/lda-illustration-blei2.png"
         alt="fp7 totalcost hist1"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The original illustration of LDA in terms of generating a document from Blei et al. (2003). 
    </div>
</div>

Now, it's the time to understand the probability distributions. For a vector $\theta_i \in \mathbb{R}^{k}$, which follows a $k$-dimensional Dirichlet distribution with parameter $\alpha$, we have:

$$
\begin{aligned}
\sum_{i=1}^{k} \theta_{i} &= 1 \\
f(\theta, \alpha) & = \frac{1}{B(\alpha)} \prod_{i=1}^{k} \theta_{i}^{\alpha_{i} - 1} \\
B(\alpha) & = \frac{\prod_{i=1}^{k} \Gamma(\alpha_{i})}{\Gamma(\sum_{i=1}^{k} \alpha_{i})}
\end{aligned} \tag{2}
$$

where $\alpha = (\alpha_1, \alpha_2, \cdots, \alpha_k)$, $\alpha_i > 0$ and $B(\alpha)$ is the normalizing constant (a multivariate generalization of the beta function).

For $N$ number of trials, the probability mass function of a multinomial distribution with parameter $\theta = (\theta_1, \cdots \theta_k)$ and $\sum_{i=1}^{k} \theta_{i} = 1$ is:

$$
f(x, \theta, N) = \frac{N!}{x_1! \cdots x_k!} \prod_{i=1}^{k} \theta_{i}^{x_{i}} \tag{3}
$$

with $x = (x_1, \cdots, x_k)$, $x_i \geq 0$ and $\sum_{i=1}^{k} x_{i} = N$.



{% endkatexmm %}