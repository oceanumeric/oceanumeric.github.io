---
title: Develop Some Fluency in Probabilistic Thinking
subtitle: The foundation of machine learning and data science is probability theory. In this post, we will develop some fluency in probabilistic thinking with different examples, which prepare data scientists well for the sexist job of the 21st century. 
layout: math_page_template
date: 2023-02-12
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality approximate counting Robert Morris hash table 
published: true
tags: probability algorithm data-science machine-learning bernoulli-trial hypergeometric-distribution
---

When I was studying probability theories, it was difficult for me
to see its application in real life until we enter a new era with 
all potential of machine learning and deep learning models. 
This post is to explain how those basic 
theories (especially those related to inequalities) could
help us for solving real world problems. Personally, I benefit
a lot from those examples because it shows me what is probabilistic thinking
and how powerful it is. Much of the content of this post is based on Cameron Musco's course - [Algorithms for Data Science](https://people.cs.umass.edu/~cmusco/CS514F22/index.html){:target="_blank"} {% cite musco2022 %} . If you want to review your knowledge about probability theory, you can 
check my post - [Probability Review](https://oceanumeric.github.io/math/probability-review){:target="_blank"}. However, I suggest you read them only when this post refers to it as it is more efficient and 
much easier to understand with the context. 

This post is example driven meaning we will use different examples to develop our fluency in probabilistic thinking. 

- [Example 1: verify a claim](#example-1-verify-a-claim)
- [Example 2: approximate counting](#example-2-approximate-counting)
- [Example 3: unit testing](#example-3-unit-testing)
- [Example 4: hash tables](#example-4-hash-tables)


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
\mathbb{E} [D] = \sum_{i, j \in [m], i < j} \mathbb{E}  [D_{i, j}] \tag{2}
$$


For any pair $i, j \in [m], i < j $, we have: 

$$
\mathbb{E}  [D_{i, j}] = P(D_{i, j} = 1) = \frac{1}{n} \tag{3}
$$

because $i, j$ were drawn _independently with replacement_. 

This gives us 

$$
\mathbb{E}  [D] = \sum_{i, j \in [m], i < j} \mathbb{E}  [D_{i, j}]= \binom{m}{2} \frac{1}{n} = \frac{m(m-1)}{2n}  \tag{4}
$$

For the set $m$ elements, we choose any two from it, say $i, j$, there must be a case which either $i < j$ or $j < i$. Therefore, $\sum_{i, j \in [m], i < j}$ gives the combination of $\binom{m}{2}$. 

__Connection to the birthday paradox.__ If there are a 130 people in this room,
each whose birthday we assume to be a uniformly random day of the 365 days in the
year, how many pairwise duplicate birthdays do we expect there are?

According to the formula (4), it should be 

$$
\mathbb{E} [D] = \binom{m}{2} \frac{1}{n} = \frac{m!}{2! (m-2)!} \frac{1}{n} = \frac{130 \times (130 -1)}{2} \frac{1}{365} 
$$


<div class='figure'>
    <img src="/math/images/birthday.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Plot of number of pairs having the same birthday
        based on the population size; notice that the trend grows 'exponentially'. 
    </div>
</div>

Now, we will plot the similar graph for our CAPTCHAS checking problem. As it is shown in Figure 2, we are not expected to see many duplicates when we draw a sample randomly with the size of 2000. 

<div class='figure'>
    <img src="/math/images/captcha.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Plot of expected number of duplicates for different sample size with $n=1,000,000$.
    </div>
</div>

With $m = 1000$, if you see __10 pairwise duplicates__ and then you 
can suspect that something is up. But __how confident__ can you be
in your test?

###  Markov's Inequality

If $X$ is a nonnegative random variable and $a > 0$, then the probability 
that $X$ is at least $a$ is at most the expectation of $X$ divided by $a$ 

$$
\mathbb{P}(X \geq a) \leq \frac{ \mathbb{E}  [X]}{a} \tag{5}
$$

It is easy to prove this. For a discrete random variable, we have

$$
\begin{aligned}
\mathbb{E} [X] = \sum \mathbb{P}(X = x) \cdot x & \geq \sum_{x \geq a} \mathbb{P}(X = x) \cdot s \\
& \geq \sum_{x \geq a} \mathbb{P}(X = x) \cdot a \\ 
& = a \cdot \mathbb{P}(X \geq a)
\end{aligned}
$$

<div class='figure'>
    <img src="/math/images/markov_prob.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> Plot of Markov's inequality with $n=1,000,000$ and a fixed sample size $m=1000$.
    </div>
</div>

With the formula (5), we can measure how confident we are when it comes to the decision that whether the service provider has a large number of
CAPTCHAS in database. As it is shown in Figure 3, The probability of
having $10$ duplicate pairs with $m=1000$ is $0.04995$, which is around 5%. Therefore, we can state that we are '95%' right when we
reject the service provider's claim that their database is of size $n=1,000,000$. 

## Example 2: approximate counting 

We are using this example to explain the difference between deterministic models
and stochastic models, which is relevant for the example 4 in the coming 
section. This section is also connected with deterministic and stochastic
optimization problems. 

Approximate Counting algorithms are techniques that allow us to count a large number 
of events using a very small amount of memory. It was invented by 
Robert Morris in 1977 and was published through his paper 
[Counting large number of events in small registers](https://www.inf.ed.ac.uk/teaching/courses/exc/reading/morris.pdf){:target="_blank"}. The algorithm uses probabilistic 
techniques to increment the counter, although it does not guarantee the 
exactness it does provide a fairly good estimate of the true value while 
inducing a minimal and yet fairly constant relative error. 

Here is the problem:
- we need to counter a large number, say $1,000,000$
- however, our register is only 8-bits, which means $2^8 = 256$
- how could count this large number of events and store the value?

The idea is very simple: instead of counting one by one, we could count 1 after
10 times of even or $\log_2(1+n)$. This means we will have a function to 
map $n$ to $X_n$ like the following one:

$$
X_n = \log_2(1 + n)
$$ 

The tradeoff is to lose accuracy to count a large number :

- $\log_2(1+0) = 0$, we start from 0 
- $\log_2(1+100) = 6.64$, we may count it as 6 or 7
- $\log_2(1+120) = 6.91$, we may still count it as 6 or 7 
- $\log_2(1+130) = 7.02$, we may now count it as 7

Therefore the accuracy rate of the above algorithm depends on how we round
our errors. We could design a rule as follows:

- round up if value of the first decimal digit is greater or equal than $5$
- round down otherwise

This kind of algorithm is called deterministic as the rule is determined pre-running and the error was also introduced deterministically. However, we
could also introduce the error stochastically as a random variable following 
some distribution. 

Now, again the value $X_n$ stored in a counter is

$$
X_n = \log_2(1 + n)
$$ 

where $n$ is the number of events that have occurred so far. The plus one
is to make sure that $n = 0 \equiv X_n = 0$. At any given time, our best
estimate of $n$ is 

$$\hat{\theta}_n = 2^{X_n} -1 $$

$\hat{\theta}_n$ is the estimation. For the next event, we have  

$$\hat{\theta}_n +1 = 2^{X_n} $$

Suppose in the midst of counting, we have the value $X_n$ stored in 
the register. Whenever we get another event, we attempt to 
modify the contents of the register in the most _appropriate_ way. The question
is: _how could we modify the contents of the register in the most appropriate way_ ?

To increment the counter, we compute

$$X_{n+1} = \log_2(1 + 2^{X_n})$$

However, values we calculated or mapped are not integer anyway. If we round 
the value, we might accumulate errors in our approximation. Instead, what we
can do is the following (by replying on probability). 

First, initialize $X_0 = 0$. Next, for $n > 0$, compute

$$p = \frac{1}{2^X}, \quad r \sim \text{Uniform}(0, 1)$$

Then the value of $X_{n+1}$ is

$$
X_{n+1} = \begin{cases}
X_n + 1 & \text{if} \  p > r \\
X_n & \text{else}
\end{cases}
$$

Before we analyze Morris' algorithm, let's compare the counting of those two 
methods: deterministic and stochastic one. 

<div class='figure'>
    <img src="/math/images/morris_count.png"
         alt="A demo figure"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Plot of two different kinds of counting algorithms: notice the deterministic one gives the same results, whereas the 
        stochastic one has the different path (comparing left and right panel). 
    </div>
</div>

As it is shown in Figure 4, every time when we run Morris' approximate counting, the result is 
different as it depends on the value of $r$. However we can show 
that the _expectation _of the approximation is $n$: 

$$
\begin{aligned}
\hat{\theta}_n & = 2^{X_n} -1  \\ 
\mathbb{E} [\hat{\theta}_n] & = n
\end{aligned}
$$

For the base case when $n = 0$, we have $ \mathbb{E} [2^{X_0} - 1] = 0$. 

Now, we will prove the above expectation by induction. With the condition that
base case is true, assume that 

$$\mathbb{E} [\hat{\theta}_n] = n$$

Then 

$$
\begin{aligned}
\mathbb{E}\left[\hat{\theta}_{n+1}\right] & \triangleq \mathbb{E}\left[2^{X_{n+1}}-1\right] \\
& =\sum_{i=1}^{\infty} \mathbb{P}\left(X_n=i\right) \mathbb{E}\left[2^{X_{n+1}} \mid X_n=i\right]-1 \\
& =\sum_{i=1}^{\infty} \mathbb{P}\left(X_n=i\right)[\underbrace{\frac{1}{2^j} \cdot 2^{j+1}}_{\text {increment }}+\underbrace{\left(1-\frac{1}{2^j}\right) \cdot 2^j}_{\text {no change }}]-1 \\
& =\sum_{i=1}^{\infty} \mathbb{P}\left(X_n=i\right)\left(2^j+1\right)-1 \\
& =\sum_{i=1}^{\infty} \mathbb{P}\left(X_n=i\right) 2^j+\sum_{i=1}^{\infty} \mathbb{P}\left(X_n=i\right)-1 \\
& =\mathbb{E}\left[2^{X_n}\right] \\
& =n+1 .
\end{aligned}
$$

Although Morris' algorithm could deviate a lot from the true value, the 
expectation is still equal to the true value. The deterministic one has 
a constant deviation but the error is accumulated for sure and we might end up with over counting or under counting when $n$ grows very large. 


## Example 3: unit testing

Unit tests are typically automated tests written and run by 
software developers to ensure that a section of an application 
(known as the "unit") meets its design and behaves as intended. In procedural 
programming, a unit could be an entire module, but it is 
more commonly an individual function or procedure.

Let's say that you have a dataset with millions of rows and hundreds of variables
as columns. Now, you need to write a function to go through each row and 
do some transformation column wise. For example, suppose you need to write
a function to separate year and month from `publn_date` in Table 1. As you 
can see that the format is different. Since the dataset is big, there is
no way to go through every cell to check the format and write a function 
to do the separate. What should we do? We rely on probability reasoning 
and sampling again. 


<div class="table-caption">
<span class="table-caption-label">Table 1.</span> A dataset example
</div>

| pat_publn_id | publn_kind | appln_id | publn_date | publn_first_grant |
|:------------:|:----------:|:--------:|:----------:|:-----------------:|
|  277148936   |     A1     |  156990  | 2008-05-14 |         N         |
|  437725522   |     B1     |  156990  | 2015-04-08 |         Y         |
|  437725789   |     A2     |  156990  | 2018,04,08 |         Y         |
|  437723467   |     B1    |  156990  | 19, July 2020 |         Y         |


Let's transform this problem as a probabilistic one:

- population size (number of rows of dataset): N
- size of anomalies: A
    - this means you function could cover $N-A$ cases 
    - but for those anomalies with size of $A$, the function will not pass the 
    unit testing
- random sample size: m
    - every time you run the unit testing, you draw a sample of size $m$ randomly
    and run the unit testing
- Question:
    - scenario i: we will only test once with sample size of $m$, we want to find
    out the optimal size of $m$ with parameters $A$ and $N$
    - scenario ii: let's fix $m=5$, we want to know how confident we are if we
    see our code passes the unit testing after $3$ trials

#### Scenario 1

For scenario i, we could model it as a hypergeometric distribution problem. Before
we solve our problem, let's review a classical example first. If we have
an urn filled with $w$ white and $b$ black balls, then drawing $m$ balls out
of the urn with _replacement_, then it yields a Binomial distribution
$\mathrm{Bin}(m, w/(w+b))$ for the number of white balls obtained in $m$ trials,
since the draws are independent Bernoulli trials, each with probability
$w/(w+b)$ of success {% cite blitzstein2015introduction %}. 

_Remark_: when I tried to model the unit testing problem, I was confused 
whether I should treat my unit testing as Bernoulli trial or not. The way of thinking
about it is to think about:
- what is an event? 
- how do you define a trial? 
- does your problem fits with Bernoulli distribution? (meaning the success
    rate and failure rate should be fixed for each trial)

For our unit testing, it is true that each unit test is a trial with 
success rate and failure rate. However, those rates are not known to us
and they are not fixed either. Therefore, we cannot model it as Bernoulli
trial. 


Now, back to our white and black ball problem, if we instead sample
_without replacement_, then it will not follow Bernoulli distribution. Therefore,
we need to do probabilistic reasoning. The population size is $N = w + b$. Let's
calculate count the number of possible ways o draw exactly $k$ white balls (assuming
$k < w$)  and $n-k$ black balls from the urn (without distinguishing between different
orderings for getting the same set of balls):

- we have $\binom{w}{k}$ ways of drawing k white balls out of $w$
- we have $\binom{N-k}{b}$ ways of drawing $N-k$ black balls 
- how many ways of drawing $k$ white balls out of $N$ then? It is 

$$ \binom{w}{k} \binom{N-k}{b} $$ 

Think about this way: you have 3 ways of cooking food A and 4 ways of cooking 
food B, how many ways of cooking food $AB$ then? 

For the total ways of drawing $m$ balls without differentiating colors, we have
$ \binom{N}{m}$. Since all samples are equally like, the naive definition
of probability gives us 

$$ \mathbb{P}(X = k) = \frac{\binom{w}{k} \binom{b}{m-k} }{\binom{N}{m}}$$

For our unit testing problem, we have the population size is $N$ and size of
anomalies is $A$ (treat them as black balls) and number of normal cells is $N-A$. If we draw
$m$ out of $N$, we want to know the probability of having $1, 2, \cdots k$ 
anomalies ($k < m $ and $A$). 

$$
\mathbb{P}(X = k) = \frac{\binom{A}{k} \binom{N-A}{m-k}}{\binom{N}{m}} \tag{6}
$$

Based on the equation (6), we will run the simulation with the following
parameters:

- case 1: $N=1000, A = 10$ (1% of population are anomalies)
- case 2: $N=1000, A = 100$ (10% of population are anomalies)

Since $k$ has to be smaller than $m$ and $A$, we will set $m \in [2, 99)$
and 

$$
k = \begin{cases}
\lfloor 0.5 \times m \rfloor, 0.5 \times m < A \\ 
A, \text{otherwise}
\end{cases}
$$

this means we are simulating the probability of having half of $m$ as anomalies whenever $0.5 \times m < A$, otherwise simulate the probability of having 
$A$ anomalies.   

<div class='figure'>
    <img src="/math/images/hypergeom_case1.png"
         alt="A demo figure"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> Simulation for the case 1, which shows the probability of having either half of the sample as 
        anomalies or all anomalies (K=10) drops dramatically, which is 5.32e-11.
    </div>
</div>


Figure 5 shows that the probability of having all anomalies in our sample 
is very small, which is $5.32e-11$, this means there is no unit test could 
cover all anomalies _once_ unless you test all population. Figure 6 tells
the same story in terms of trend but the probability is higher as the 
share of anomalies is higher too. 


<div class='figure'>
    <img src="/math/images/hypergeom_case2.png"
         alt="hypergeom simulation case2"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 6.</span> Simulation for the case 2, which shows the same trend but the probability grows with a larger share
        of anomalies. 
    </div>
</div>

Our simulation tells us that no one should try to come up a unit test to 
cover all of anomalies unless one is willing to test all. If we test 
based on the sample, then the optimal size is $3$, which shows the probability
of having one 1 anomalies is around $3\%$ when the share of anomalies is
$1\%$, that is around $25\%$ when the share of anomalies is $10\%$. 


#### Scenario 2 

Now, we will fix the number of anomalies, and study the probability of having exact $1$ anomalies in our sample when the parameters of interest, $N, A, m$ are changing. 

<div class="table-figure-wrapper" markdown="block">

|  m |   Probability |
|:----------:|:----:|
| 5 |  0.048 |
| 5 |  0.329 |
| 5  |  0.411 |
|25 | 0.201 |
|25 | 0.198 |
|25 | 0.023|


<div class='sub-figure'>
    <img src="/math/images/hypergeom_scenario2.png"
         alt="hypergeom scenario 2 simulation"
         style="width: 75%; display: block; margin: 0 auto;"/>
</div>

<div class='caption'>
        <span class='caption-label'>Figure 6.</span> The hypergeometric simulation when m is fixed and N and A are updated
</div>
</div>

Figure 6 shows that the probability of having exact one anomaly in
our test increases first and then drops. The optimal testing size $m$
also shifts to the left when the share of anomalies in the population
is higher, meaning we don't have to test that much to see an anomaly in our
sample. 

#### Bernoulli trial

Now, we can use Bernoulli trial as have our binary probability. Let's assume
that $N=1000, A=10, m=5$, which means the probability of having exact one
anomaly in our sample is around $0.048$ as it is shown in Figure 6. Now,
the probability of having no anomaly is:

$$
\begin{aligned}
\mathbb{P}(k=0) & = 1 - \mathbb{P}(k=1) - \mathbb{P}(k=2) - \cdots \mathbb{P}(k=5) \\ 
& \approx 1 - 0.049 = 0.951 
\end{aligned}
$$

#### Binomial trials 

Therefore, we can set $p=0.049, q = 0.951$ and if we run our unit testing three
times or five times, the probability that we will see one anomaly from any of those three
tests is

$$
\mathbb{P(u=3)} = \binom{3}{1} p (1-p)^2 \approx 0.132; \quad \mathbb{P(u=5)} = \binom{5}{1} p (1-p)^4 \approx 0.20; 
$$

If you look at the table in the figure 6, running 5 times unit testing with 
sample size $m=5$ is no difference from running one time unit testing
with sample size $m=25$. 

<div class='figure'>
    <img src="/math/images/distribution_chart.gif"
         alt="relationship of distributions"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 7.</span> The relationship of
        different distributions (as a reflection of example 3); figure was
        taken from the website of John D. Cook, PhD.
    </div>
</div>


## Example 4: hash tables 

This section assumes readers know what hash table is. If not, just check out 
[this video](https://youtu.be/Nu8YGneFCWE){:target="_blank"} from MIT. 

The hash function $h: U \to [n]$ maps the elements of universe or population to 
indices $1, \cdots, n$ of an array. The data structure `dict` from `Python` is built on hash tables as this kind of map makes it easy to extract information. 

<div class='figure'>
    <img src="/math/images/hash_table_dict.png"
         alt="A demo figure"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 8.</span> Illustration of hash table for dict structure of Python. 
    </div>
</div>

Why could not use array to build the dictionary, one array for keys and one array for values and both arrays are mapped with the same index. If we implement this idea, then we need to figure out a way of extracting the 
value based on the key, say `dict_foo["B"]`. If all keys are numbers, then
dictionary is equivalent to array. However, very often keys are strings. How
could achieve $O(1)$ performance for finding `dict_foo["B"]`? A hash table 
could do that by mapping keys into the indexes of an array ([for further explanation](./pdf/lecture4_hashing.pdf){:target="_blank"}). 

A hashing function could be either deterministic one (`Python` implemented it) or stochastic one. We will focus on the stochastic one. 

For a big dataset of $|U| >> n$, we need to reply on probability to
map elements from universe to the array as the it is economically efficient. For instance, big firms like Google or Facebook are always trying to use less servers or database to serve
their customs. In this case the population size is much larger than  the number of servers or database.  

<div class='figure'>
    <img src="/math/images/ip_hash_table.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 9.</span> Illustration of hash table for IP addresses, figure was taken from Musco's course <a href="#musco2022"> (2022)</a>. 
    </div>
</div>

As we have explained that it costs a lot to have an index for 
each element shown in Figure 9. To use less indexes in our
hash table, we need to find a way to avoid collisions. 

_Remark:_ the application of hashing in data science and data structure like `dict` is slightly different. For building a data structure like `dict`,
the focus is on achieving $O(1)$ performance of finding the value based on the key, whereas for managing a large number of visiting to server the focus is more on mapping between $m$ and $n$ dynamically.  

Now, we store $m$ items from a large universe in a hash table with
$n$ positions. When we insert $m$ items into the hash table we may
have to store multiple items in the same location (typically as a
linked list).

Let's frame our problem as a probabilistic one:
- $m$: total number of stored items (sample size)
- $n$: hash table size (population size)
- $x_i, x_j$: any pair of store items chosen randomly
- $C$: total pairwise collisions
- $h$: random hash table function 

_Remark:_ In example 1, the population size $n=1,000,000$ is fixed, whereas the sample size $m$ is updated dynamically every time we 
draw a sample; for example 2, the hash table size $n$ is fixed too, whereas total number of stored items is updated dynamically. It is important to know when and how to set up _population_ and _sample space_. _It might be counterintuitive when you see we set the hash table as the population_. 


__General principle for setting up population and sample__: Anything is fixed should be treated as the population, whereas anything is updated dynamically should be treated as the sample space that could be mapped with the random variable function. 

With the right setup, we can have the following expectation of pairwise
collision 


$$
\mathbb{E}  [C] = \sum_{i, j \in [m], i < j} \mathbb{E}  [C_{i, j}]= \binom{m}{2} \frac{1}{n} = \frac{m(m-1)}{2n}  \tag{7}
$$

For $n = 4m^2$ (size of hashing table is much larger than the sample
size), we have 

$$
\mathbb{E} [C] = \frac{m(m-1)}{8m^2} < \frac{m^2}{8m^2} = \frac{1}{8} \tag{8}
$$

This means we could have the collision-free hash table if the hash
table is large enough. This completely makes sense.  However,
we do not gain any economic benefit as we are using $O(m^2)$ space
to store $m$ elements. 

Let's apply Markov's inequality in this case, we can have

$$
\begin{aligned}
\mathbb{P} (C \geq 1) \leq \frac{\mathbb{E} (C)}{1} = \frac{1}{8} \\ 
\mathbb{P}(C = 0) = 1 - \mathbb{P} (C \geq 1) = \frac{7}{8} \tag{9}
\end{aligned}
$$

__Two Level Hashing.__ Now, if we want to preserve $O(1)$ query with 
$O(m)$ space, we can do two level hashing as shown in Figure 5. 


<div class='figure'>
    <img src="/math/images/two_level_hashing.png"
         alt="A demo figure"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 10.</span> Illustration of two level hashing; figure was taken from Musco's course <a href="#musco2022"> (2022)</a>. 
    </div>
</div>

In equation 9, we have shown that the random hashing function could 
guarantee that the collision is avoided at $7/8$ probability level. Therefore, we could just use a random hashing function at level two. Now, let's set up the scene to do probabilistic thinking:
- $m$: total number of stored items (sample size)
- $n$: hash table size (population size)
- $x_j, x_k$: any pair of store items chosen randomly
- $C$: total pairwise collisions
- $h$: random hash table function 
- $S$: spaced use for the first and second level of hashing 
- $s_i$: items (plural) stored in the hash table at position $i$ (see Figure 5)
    - for each bucket in hashing table of size $n$, pick a collision
    free hash function mapping $[s_i] \to [s_i^2]$ (as it is shown in equation 8). 

What is the expected _space usage_ for two level hashing? It is quite straightforward
to calculating the total space usage.

$$
\mathbb{E} [S] = n + \sum_{i=1}^n \mathbb{E} [s_i^2] \tag{10}
$$

Before we calculating $\mathbb{E} [s_i^2]$, let's go through two scenario (as always, we 
assumed $m > > n$):

- determined path: for $m$ items, we assign $n$ of them into our hashing table
and then we assign $(m-n)$ elements into the second level linked list
- stochastic path: every item was assign to a cell randomly and we will use an indicator function to _track_ them. 
    - we don't know which element will be assigned into the container $s_i$ ($i$ is the index of position)
    - but we could use an indicator function to track them 

$$
\mathbb{I}_{\{s_i\}}[h(x_j)] := \begin{cases}  1 & h(x_j) = i \\
0 & h(x_j) \neq i
\end{cases}
$$
 

Therefore, we could have 

$$
\mathbb{E}[s_i^2] = \mathbb{E} \left [ \left ( \sum_{j=1}^m \mathbb{I}_{h(x_j)=i}  \right )^2 \right] \tag{11}
$$

Equation (11) is saying that we sum up all possible elements assigned into
$s_i$ for element $x_j, j \in [m]$. Now, let's expand the equation 11,

$$
\begin{aligned}
\mathbb{E}[s_i^2] & = \mathbb{E} \left [ \left ( \sum_{j=1}^m \mathbb{I}_{h(x_j)=i}  \right )^2 \right]  \\
& = \mathbb{E} \left [ \left ( \sum_{j=1}^m \mathbb{I}_{h(x_j)=i}  \right ) \left ( \sum_{j=1}^m \mathbb{I}_{h(x_j)=i}  \right ) \right] \\ 
& = \mathbb{E} \left[ \sum_{j, k \in [m]}  \mathbb{I}_{h(x_j)=i}  \cdot \mathbb{I}_{h(x_k)=i} \right ]  \\
& = \sum_{j, k \in [m]} \mathbb{E} \ \Bigl[\mathbb{I}_{h(x_j)=i} \cdot \mathbb{I}_{h(x_k)=i} \Bigr] \tag{12} 
\end{aligned}
$$

The second step of equation (12) is derived because many values of indicator
function are zeros and the last step is derived due to the linearity of expectation. 

With equation (12), we need to calculate the expectation

$$\sum_{j, k \in [m]} \mathbb{E} \ \Bigl [\mathbb{I}_{h(x_j)=i} \cdot \mathbb{I}_{h(x_k)=i} \Bigr ] $$

We will do this by calculating two cases:
- $ j = k$
- $j \neq k$ 

For $j=k$, the derivation is simple 

$$
\mathbb{E} \ \Biggl [\mathbb{I}_{h(x_j)=i} \cdot \mathbb{I}_{h(x_k)=i} \Biggr] = \mathbb{E} \left[ \left (\mathbb{I}_{h(x_j)=i} \right)^2 \right] = \mathbb{P}[(h(x_j) = i)] = \frac{1}{n}
$$

For $j \neq k$, we have 


$$
\mathbb{E} \ \Biggl [\mathbb{I}_{h(x_j)=i} \cdot \mathbb{I}_{h(x_k)=i} \Biggr] = \mathbb{P}[(h(x_j) = i) \cap h(x_k) = i] = \frac{1}{n^2}  
$$

Therefore, according to the linearity of expectation, we have

$$
\begin{aligned}
\mathbb{E}[s_i^2] & = \sum_{j, k \in [m]} \mathbb{E} \ \Bigl[\mathbb{I}_{h(x_j)=i} \cdot \mathbb{I}_{h(x_k)=i} \Bigr] \\
& = m \cdot \frac{1}{n} + 2 \cdot \binom{m}{2} \frac{1}{n^2} \\
& = \frac{m}{n} + \frac{m(m-1)}{n^2} \tag{13}
\end{aligned}
$$

When we set $n = m$ in equation (13), we can get

$$
\mathbb{E}[s_i^2] = \frac{m}{n} + \frac{m(m-1)}{n^2} < 2  \tag{14}
$$

Now, with this bound, we can substitute the equation (14) into equation (10) and have our final result,

$$
\mathbb{E} [S] = n + \sum_{i=1}^n \mathbb{E} [s_i^2] \leq n + \sum_{i=1}^n 2 = 3n = 3m \tag{15}
$$

This means we achieved a near optimal space with $O(1)$ query time! 


{% endkatexmm %}