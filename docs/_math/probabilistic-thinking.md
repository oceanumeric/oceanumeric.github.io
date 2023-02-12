---
title: Develop Some Fluency in Probabilistic Thinking
subtitle: The foundation of machine learning and data science is probability theory. In this post, we will develop some fluency in probabilistic thinking with different examples, which prepare data scientists well for the sexist job of the 21st century. 
layout: math_page_template
date: 2023-02-12
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality 
published: true
tags: probability algorithm data-science machine-learning
---

When I was studying probability theories, it was difficult for me
to see its application in real life until we enter a new era with 
all potential of machine learning and deep learning models. 
This post is to explain how those basic 
theories (especially those related to inequalities) could
help us for solving real world problems. Personally, I benefit
a lot from those examples because it shows me what is probabilistic thinking
and how powerful it is. Much of the content of this post is based on Cameron Musco's course - [Algorithms for Data Science](https://people.cs.umass.edu/~cmusco/CS514F22/index.html){:target="_blank"}. If you want to review your knowledge about probability theory, you can 
check my post - [Probability Review](https://oceanumeric.github.io/math/probability-review){:target="_blank"}. However, I suggest you read them only when this post refers to it as it is more efficient and 
much easier to understand with the context. 

This post is example driven meaning we will use different examples to develop our fluency in probabilistic thinking. 

## Example 1: verify a claim

{% katexmm %}

You have contracted with a new company to provide [CAPTCHAS](https://en.wikipedia.org/wiki/CAPTCHA){:target="_blank"} for
your website. They claim that they have a database of 1,000,000 
unique CAPTCHAS. A random one is chosen for each security check.
You want to independently verify this claimed database size.
You could make test checks until you see all unique CAPTCHAS, which 
would take more than 1 million checks. 

- A probabilistic solution: You run some test security checks and see if any duplicate
CAPTCHAS show up. If you’re seeing duplicates after not too many
checks, the database size is probably not too big. 
- we have:
    - $n$: number of CAPTCHAS in database
    - $m$: number of random CAPTCHAS drawn to check database size
    - $D$: number of pairwise duplicates in $m$ random CAPTCHAS
    - For test $i, j \in m$, if tests $i$ and $j$ give the same CAPTCHA (meaning
    we have duplicate ones), let $A$ as the set of duplicates, 
    we could set an indicator variable 
        $$ D_{i, j}:={\begin{cases}1~&{\text{ if }}~i, j\in A~,\\0~&{\text{ if }}~i, j\notin A~.\end{cases}} $$
        
Then the number of pairwise duplicates is 

$$
D = \sum_{i, j \in[ m], i < j} D_{i, j} \tag{1}
$$

Then we can have 

$$
E [D] = \sum_{i, j \in [m], i < j} E [D_{i, j}] \tag{2}
$$


For any pair $i, j \in [m], i < j $, we have: 

$$
E [D_{i, j}] = P(D_{i, j} = 1) = \frac{1}{n} \tag{3}
$$

because $i, j$ were drawn _independently with replacement_. 

This gives us 

$$
\mathrm{E} [D] = \sum_{i, j \in [m], i < j} \mathrm{E} [D_{i, j}]= \binom{m}{2} \frac{1}{n} = \frac{m(m-1)}{2n}  \tag{4}
$$

For the set $m$ elements, we choose any two from it, say $i, j$, there must be a case which either $i < j$ or $j < i$. Therefore, $\sum_{i, j \in [m], i < j}$ gives the combination of $\binom{m}{2}$. 

__Connection to the birthday paradox.__ If there are a 130 people in this room,
each whose birthday we assume to be a uniformly random day of the 365 days in the
year, how many pairwise duplicate birthdays do we expect there are?

According to the formula (4), it should be 

$$
E[D] = \binom{m}{2} \frac{1}{n} = \frac{m!}{2! (m-2)!} \frac{1}{n} = \frac{130 \times (130 -1)}{2} \frac{1}{365} 
$$

```python
def birthday(m:int) -> int:
    """
    Calculate the number of pairs having duplicate birthdays
    """
    return int(m * (m-1) / (2 * 365))
```


<div class='figure'>
    <img src="/math/images/birthday.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Diagram of rejection sampling. The 
        density $q(\mathbf{z})$ must be always greater than $p(\mathbf{x})$. A new sample 
        is rejected if it falls in the gray region and accepted otherwise. These accepted 
        samples are distributed according to $p(\mathbf{x})$. This is achieved by sampling 
        $z_i$ from $q(\mathbf{z})$, and then sampling uniformly from $[0, k q(z_i)]$. 
        Samples under the curve $p(z_i)$ are accepted.
    </div>
</div>







{% endkatexmm %}