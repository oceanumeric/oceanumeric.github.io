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
& & M \times V & & \\
& & & & \\
& & & &
\end{bmatrix}}_{\text{Corpus}} \tag{1}
$$

The original paper by Blei et al. did not use the above matrix representation. But they did give the following graphical representation, which is very helpful to understand the generating process of LDA. 

<div class='figure'>
    <img src="/images/blog/lda-illustration-blei2.png"
         alt="fp7 totalcost hist1"
         style="width: 65%; display: block; margin: 0 auto;"/>
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


## The estimation of LDA

Before we estimate the parameters of LDA, let's walk through the generative process again with our probability distributions. I found it easier to understand the generative process with probability distributions. 

Our goal is to choose a combination of topics and words that can best explain the corpus. First, we set up a key assumption: __the number of topics $k$ is known and fixed before we start the estimation__. Each topic is a mixture of words, which will be sampled to construct a document. 


For $k$ topics, we assign weights $\theta_{ij}$ to each topic, where $i$ is the index of document and $j$ is the index of topic. 

$$
\begin{aligned}
d_i & = \text{document}_i  \ \ \text{with weights of topics} \\
    & = [\theta_{i1}, \theta_{i2}, \cdots, \theta_{ik}] \\
    & = [0.2, 0.07, \cdots, 0.03]
\end{aligned} \tag{4}
$$

Equation (4) gives the weight of topics in document $i$. It looks if we have weights of topics in a document, we can generate a document by sampling words from the topics. Figure 4 shows the process. 

<div class='figure'>
    <img src="/images/blog/lda-illustration-blei3.png"
         alt="fp7 totalcost hist1"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The illustration of LDA in terms of generating a document from Blei et al. (2003). We assume there are 4  topics in this illustration.
    </div>
</div>


As it has posed in figure 4, what are the key assumptions we set up in the above process?
To generate a document and corpus, we need to know:

1. the weights of topics in a document
2. the number of words in each topic

When I read many blogs and papers about LDA, I found that most of them did not explain the above two assumptions, especially the second one. Most blogs and papers only explain the first assumption, which is easy to understand. 

For the weights of topics in a document, we can use the Dirichlet distribution to sample the weights. This means we have our prior knowledge about the weights of topics in a document as a Dirichlet distribution.

$$
\theta \sim \text{Dir}(\alpha) \tag{5}
$$


But how about the number of words in each topic? We can use the multinomial distribution to sample the number of words in each topic. This means we have our prior knowledge about the number of words in each topic as a multinomial distribution. That's why we need to set up the hyperparameter $\beta$ in the beginning.

Now, we need to come up a mechanism to determine the number of words in each topic. Suppose we have $V$ words in our corpus, we will assign weights $\beta_{kv}$ to each word, where $k$ is the index of topic and $v$ is the index of word. This could be done by sampling from a Dirichlet distribution. I don't know why authors of the original paper did not mention this.


<div class='figure'>
    <img src="/images/blog/lda-illustration-blei4.png"
         alt="fp7 totalcost hist1"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The illustration of LDA in terms of generating a document from Blei et al. (2003). We assume there are 4  topics in this illustration.
    </div>
</div>

Suppose $V = 5$, and $k=4$, we can simulate the $\beta$ matrix as follows. 

```R
beta_hyperparameter <- c(1,2,3,2.7,9)
rdirichlet(4, beta_hyperparameter)
# 0.039350115	0.20518288	0.1656238	0.19104531	0.3987978
# 0.001968719	0.11850665	0.1971715	0.13461901	0.5477341
# 0.109902876	0.06116357	0.2768535	0.07761342	0.4744666
# 0.039633147	0.25311539	0.1254669	0.17837372	0.4034108
```

The each row gives the weight of words in a topic. 


Now, let's walk through the generative process again. We have the weights of topics in a document as $\theta_d$, which is a $1 \times k$ vector. Then we will construct a $k \times V$ matrix $\beta$. To construct this matrix, we could do it two ways:

1. for each word assign weights to each topic, and then repeat the process for all words - $V$ times
2. for each topic assign weights to each word, and then repeat the process for all topics - $k$ times

Authors of the original paper chose the second way. When we compute the probability of a word in the document, we just need to multiply the weights of topics in a document and the weights of words in a topic, such as $\theta_{d} \times \beta$. 


There over $k$ topics, the probability of a word in a document is:

$$
P(w_{d,n} | \theta_{d}, \beta) = \sum_{k=1}^{k} \theta_{d,k} \times \beta_{k,w_{d,n}} \tag{6}
$$

Equation (6) is just the vector multiplication of $\theta_{d}$ and $\beta_{w_{d,n}}$, where $\theta_{d}$ is $1 \times k$ vector and $\beta_{w_{d,n}}$ is $k \times 1$ vector.

Now, with $N$ number of words in a document, we can have the probability of a document as:

$$
P(d_{i} | \theta_{i}, \beta) = \prod_{n=1}^{N} P(w_{d,n} | \theta_{d}, \beta) \tag{7}
$$

We can use the above equation to compute the probability of a document. But how can we estimate the parameters $\theta_{d}$ and $\beta$? We can use the maximum likelihood estimation to estimate the parameters.

$$
\begin{aligned}
\hat{\theta}_{d} & = \text{argmax}_{\theta_{d}} P(d_{i} | \theta_{i}, \beta) \\
\hat{\beta} & = \text{argmax}_{\beta} P(d_{i} | \theta_{i}, \beta) \\
\end{aligned} \tag{8}
$$

Now, we can estimate the posterior distribution of $\theta_{d}$ and $\beta$ by using the Gibbs sampling method. 















{% endkatexmm %}