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

One way to calculate similarity is to calculate the share of
intersections among two sets. 

Consider two sets $A=\{0,1,2,5,6\}$ and $B=\{0,2,3,5,7,9\}$. How similar are $A$ and $B$ ? The Jaccard similarity is defined
$$
\begin{aligned}
\mathrm{JS}(A, B) & =\frac{|A \cap B|}{|A \cup B|} \\
& =\frac{|\{0,2,5\}|}{|\{0,1,2,3,5,6,7,9\}|}=\frac{3}{8}=0.375
\end{aligned}
$$

Suppose we want to find the similar documents or names, we could use Jaccard similarity. However, we need to construct 
the sample space $\Omega$ and samples first:

- use $n$-Grams technique to construct a set for each document as a field $\mathcal{F}$
- the union of all sample sets is sample space $\Omega$ 
- calculate the share of intersection pairwise 

### naive method

Now, suppose we have four documents as four sets 

$$
\begin{aligned}
S_1 & = \{1, 2, 5 \} \\
S_2 & = \{ 3 \} \\
S_3 & = \{ 2, 3, 4, 6 \} \\
S_4 & = \{1, 4, 6 \}
\end{aligned}
$$

and we will construct
one-hot vector for each document 

|Index | Element | $S_1$ | $S_2$ | $S_3$ | $S_4$ |
|:---:| :---: | :---: | :---: | :---: | :---: |
|1| hel | 1 | 0 | 0 | 1 |
| 2| ell | 1 | 0 | 1 | 0 |
| 3 |llo | 0 | 1 | 1 | 0 |
| 4 |o w | 0 | 0 | 1 | 1 |
| 5 |wor | 1 | 0 | 0 | 0 |
| 6 | orl | 0 | 0 | 1 | 1 |


When both sets have $1$ for each row, it means that they 
have intersections. It is very intuitive. It looks like we do not need to use hash function. However, what we did is

- construct a matrix first
- select the pair we want to compare, such as $S_1, S_3$
- calculate the true (if both true) or false value pairwise

Let's calculate the running time for this process:

- load (or read in) the dataset
- prepare the sets by doing n-grams and sample space (population)
- construct the matrix
- double loop for calculating the share, (n choose 2) = O($n^2$)

You can see that it requires lots of calculations. Furthermore, reading all elements of n-grams and constructing a matrix could already takes a huge amount of time if you have a big dataset. Therefore, we need a 
random process to reduce the running time and space.

Here is the pseudo code for the native method to finding similar items.

```python
# construct samples and save them into a dict 
doc_dict = {}
for i in range(number_of_files):
    doc = read_file(f"file_name_{i}") 
    n_grams_i = construct_n_grams(doc)
    doc_dict["file_i"] = n_grams_i

# sample space (population) by get the union
doc_union = []
for sample in doc_dict.values():
    doct_union.append(sample)

doc_union = set(doc_union)

# construct the matrix 
n = len(doc_union)
m = len(doc_dict)
jacarrd_matrix = np.zeros((n, m))

# loop over all samples (sets) and do one-hot technique
for s, idx in enumerate(doc_dict.values()):
    one_hot = np.isin(doc_union, s).reshape(-1, 1)
    jacarrd_matrix[:, idx] = one_hot

# calcuate the similarity
idx_list = list(range(m))
for i in range(m):
    pair_list_idx = set(idx_list).diff(set(i))
    for j in pari_list_idx:
        # calcuate the share
        share = np.sum(jacarrd_matrix[:, i] + jaccard_matrix[:, j])
# hope you can see how naive this method is 
```

It could run days if you have a large amount of documents, and even years when you have a big dataset. 


## Min hashing algorithm 

The idea of min hashing is to do everything we did in the above section
with one pass, which means we want to get everything at once when we load the dataset. 

Instead of calculating the full vector, we use $k$ hash functions to create a signature of each document and then compare those signatures pairwise to find similar documents. The intuition of this idea 
is similar with that of finding the distinct values (read this
[post](https://oceanumeric.github.io/math/2023/03/approximating-distinct-elements#algorithms-in-practice)). 


This means we will do two passes. Is it possible to calculate Jaccard similarity with only _one pass_? YES, it is! To do
this, we need our favorite _random process_.

<div class='figure'>
    <img src="/math/images/min-hashing.png"
         alt="Inequality bounds compare"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Illustration of Minimum Hashing process.
    </div>
</div>

As it is shown in Figure 1, we will create n-Grams for each documents when 
we read them in and then hash them with $k$ hash functions. For each batch of n-Grams of each document, we choose the minimum value among them and save 
it into a signature vector of length $k$. 

Once we have the signature vector, we could calculate the Jaccard similarity
pairwise. 

Compare to the native method, what did we gain from MinHash algorithm?

|                                          | native method    | MinHash method  |
|:------------------------------------------|------------------|-----------------|
| load the dataset                         | M                | M               |
| construct n-Grams sets                   | n                | n               |
| encode each set as one-hot vector        | $\footnotesize n \times  N$     | 0               |
| calculate Jaccard similarities pairwise | $\footnotesize (M \cdot N )^2$ | $\footnotesize (M \cdot k)^2$ |


As you can see, we save lots of space for one-hot vector encoding part and we 
also save the running time for calculating Jaccard similarities. Since $N$ is 
the population size, the saved running time is a big number. 


However the running time for calculating the share of intersection still
grows with $O(M^2)$. Can we do better with smart algorithm? The answer is YES!

We can if we use Locality Sensitive Hashing (LSH) algorithm. Wouldn't it being
amazing if we could calculate Jaccard similarities within $O(M \cdot k)$? I
hope you could appreciate the invention of LSH and understand now why they 
won ACM Paris Kanellakis Theory and Practice Award.

## Locality sensitive hashing 


{% endkatexmm %}