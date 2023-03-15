---
title: Locality Sensitive Hashing (LSH)
subtitle: LSH is recognized as a key breakthrough that has had great impact in many fields of computer science including computer vision, databases, information retrieval, machine learning, and signal processing. 
layout: math_page_template
date: 2023-03-15
keywords: probabilistic-thinking approximating distinct elements hashing similarity
published: true
tags: probability algorithm data-science machine-learning bloom-filter minhash LSH 
---

In 2012, minwise hashing and locality sensitive hashing (LSH) were recognized as a key breakthrough and inventors were awarded ACM Paris Kanellakis Theory and Practice Award. Those inventors were awarded for "their groundbreaking work on locality-sensitive hashing that has had great impact in many fields of computer science including computer vision, databases, information retrieval, machine learning, and signal processing". 

In this post, we will learn those brilliant ideas behind LSH. Before we proceed, let's have a prelude on probabilistic reasoning. 


## Probabilistic thinking again

{% katexmm %}

I have several posts to discuss the essence of probabilistic reasoning.
If you have not read them, I recommend reading them first as this post is the climax of probabilistic thinking. 

When we talk about probability, we are talking about the following
three things:

- sample space $\Omega$, which corresponds to _population_ in statistics
- field $\mathcal{F}$, which corresponds to _samples_ in statistics
- a probability measure $\mathbb{P}$ on $(\Omega, \mathcal{F})$, which corresponds to _likelihood_ in statistics

When we think in the way of probability, our logic follows the order as such

$$
\text{sample space} \rightarrow \text{field} \rightarrow \text{probability}; \quad \Omega \to \mathcal{F} \to \mathbb{P}
$$

When we think in the way of statistics, our logic follows the order 
as such 

$$
\text{likelihood (model)} \rightarrow  \text{samples (data)} \rightarrow \text{population}; \quad  \mathbb{P} \to \mathcal{F} \to \Omega
$$

One follows the data generating process, the other follows the data inference process. 


In data science, those two logics very often entwined. However, if we 
keep the key concepts in our mind, we will not lost. The key concepts in data science are: population (cardinality), samples (events), and
probability (function mapping). 

When you are dealing with big data, I encourage you to think dynamically
which means you should think hard on how you can transfer a static 
problem into a random process (data generating) problem. 

Okay, enough talking on philosophy of probability, let's see some real world problems. 

## Similarity issue 

In the field of data science, many problems are related to calculating
similarities of different items (such as documents, or firm names). For instance, we are interested in: given two homework assignments (reports) how can a computer detect if one is likely to have been
plagiarized from the other without understanding the content? 

One way to calculate similarity is to 









{% endkatexmm %}