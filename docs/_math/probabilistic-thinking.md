---
title: Develop Some Fluency in Probabilistic Thinking
subtitle: The foundation of machine learning and data science is probability theory. In this post, we will develop some fluency in probabilistic thinking with different examples, which prepare data scientists well for the sexist job of the 21st century. 
layout: math_page_template
date: 2023-02-12
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality 
published: true
tags: probability algorithm data-science machine-learning
---


This post is very long. Therefore, we need a roadmap to guide us.
As it is shown in Figure 1, we will review some basic probability 
theories and prove some of them, then we will learn how those basic 
theories (especially those related to inequalities) could
help us to build models for solving real world problems, such as
algorithms for data science or machine learning models. Personally, I benefit
a lot from those examples because it shows me what is probabilistic thinking
and how power it is. 

## Probability review

{% katexmm %}

To treat probability rigorously, we define a _sample space_ $S$ whose elements 
are the possible outcomes of some process or experiment. For example, 
the sample space might be the outcomes of the roll
of a die, or flips of a coin. To each element $x$ of the sample space,
 we assign a probability, which
will be a non-negative number between 0 and 1, which we will denote by $\mathbb{P}(x)$.
We require that

$$\sum_{x \in S} \mathbb{P}(x) = 1$$


Let $A$ be an event with non-zero probability. We define the probability of an event $B$ conditioned
on event A, denoted by $\mathbb{P}(B | A)$, to be


$$\mathbb{P}(B | A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)}$$


If we have two events $A$ and $B$, we say that they are independent if the probability that both
happen is the product of the probability that the first happens and the probability that the second
happens, that is, if

$$\mathbb{P}(A \cap B) = \mathbb{P}(A) \cdot \mathbb{P}(B)$$ 

__Bayes's rule.__ For any two events $A$ and $B$, one has

$$\mathbb{P}(B | A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)} =   \frac{\mathbb{P}(A| B) \mathbb{P}(B)}{\mathbb{P}(A)} $$


Suppose we have a false positive rate:

$$\mathbb{P}(\text{positive test} | \text{no disease}) = \frac{1}{30}$$ 

and some false negative rate is 

$$\mathbb{P}(\text{negative test} | \text{ disease}) = \frac{1}{10}$$ 

The incidence of the desease is $1/1000$

$$\mathbb{P}(B) = 0.001$$


Let's define:

- $A$: testing positive
- $B$: having the disease 

Then we can have

$$
\begin{aligned}
\mathbb{P}(B | A) = \frac{\mathbb{P}( A \cap B)}{\mathbb{P}(A)} & = \frac{\mathbb{P}(A | B) \mathbb{P}(B) }{\mathbb{P}(A)} \\ 
& = \frac{\mathbb{P}(A | B) \mathbb{P}(B) }{\mathbb{P}(A|B)\mathbb{P}(B) + \mathbb{P}(A | \neg B )\mathbb{P}(\neg B)}  \\
& \approx 0.0265
\end{aligned}
$$


__Inclusion–exclusion principle.__ In combinatorics, a branch of mathematics, 
the inclusion–exclusion principle is a counting technique which generalizes the 
familiar method of obtaining the number of elements in the union of two finite 
sets; symbolically expressed as

$$|A \cup B| = |A| + |B| - |A \cap B |$$

The inclusion-exclusion principle, being a generalization of the two-set case, is perhaps more clearly seen in the case of three sets, which for the sets A, B and C is given by

$$ 
\begin{aligned}
|A\cup B\cup C|=|A|+|B|+|C|- & |A\cap B|-  |A\cap C| \\
    & -|B\cap C|+|A\cap B\cap C|
\end{aligned}
$$

<div class='figure'>
    <img src="/images/math/inclusion-exclusion.png"
         alt="A demo figure"
         style="width: 40%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Illustration of inclusion-exclusion principle. 
    </div>
</div>

In its general formula, the principle of inclusion–exclusion states that for finite sets A1, …, An, one has the identity

$$ 
\begin{aligned}
\left|\bigcup _{i=1}^{n}A_{i}\right|=\sum _{i=1}^{n}|A_{i}|-\sum _{1\leqslant i<j\leqslant n}|A_{i}\cap A_{j}|+ & \sum _{1\leqslant i<j<k\leqslant n}|A_{i}\cap A_{j}\cap A_{k}|- \\
& \cdots +(-1)^{n+1}\left|A_{1}\cap \cdots \cap A_{n}\right| 
\end{aligned}
$$






{% endkatexmm %}

