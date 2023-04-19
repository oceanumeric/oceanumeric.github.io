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