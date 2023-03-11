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

- [Motivation problem](#motivation-problem)
- [Probabilistic reasoning](#probabilistic-reasoning)
- [Median trick](#median-trick)
- [Algorithms in practice](#algorithms-in-practice)


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

Now, we will derive the variance for random variable $X$ with the following formula,

$$
\mathrm{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

First, we set random variable $Y = X^2$ and derive its CDF and PDF. 

$$
\begin{aligned}
F_Y(y) = \mathbb{P}(Y \leq y) & = \mathbb{P}(X^2 \leq y) \\
& = \mathbb{P}(-\sqrt{y} \leq X \leq \sqrt{y}) \\
& = \mathbb{P}(0 \leq X  \leq \sqrt{y}) \\
& = 1 - (1-\sqrt{y})^m \quad (\text{using equation (3)})
\end{aligned}
$$

Take the derivative, we can have the CDF,

$$
f_Y(y) = \frac{m}{2} (1- y^{1/2})^{m-1} y^{-1/2} \tag{6}
$$

Therefore,

$$
\begin{aligned}
\mathbb{E}[Y] & = \int_0^1 y f_Y(y) dy \\
            & = \int_0^1 y \frac{m}{2} (1- y^{1/2})^{m-1} y^{-1/2} dy \\
            & = \frac{m}{2} \int_0^1 y^{1/2} (1- y^{1/2})^{m-1} dy \\
            & = m \int_0^1 x^2 (1-x)^{m-1} dx \quad (y = x^2) \\
            & = -m \int_1^0 (1-u)^2 u^{m-1} du \quad (u = 1-x) \\
            & = \frac{2}{(m+1)(m+2)} \tag{7}
\end{aligned}
$$

Then we can have

$$
\mathrm{Var}[X] = \frac{2}{(m+1)(m+2)} - \frac{1}{(m+1)^2} = \frac{m}{(m+1)^2(m+2)} \tag{8}
$$

Therefore, we can state

$$
\mathrm{Var}[X] < \frac{1}{(m+1)^2} \tag{9}
$$

With the Chebyshev's inequality, we can show

$$
\mathrm{Pr}(|X - \mathbb{E}[X]| \geq \epsilon \mathbb{E}[X]]) \leq \frac{\mathrm{Var}[X]}{(\epsilon \mathbb{E}[X])^2}  < \frac{1}{\epsilon^2}
$$

This means the bound is vacuous for any $\epsilon < 1$. Therefore, this 
method is not very accurate. How could improve its accuracy? We can leverage the law of large numbers. 


If $X_1, \cdots, X_n$ are iid RVs with mean $\mu$ and variance $\sigma^2$, we'll show that the sample mean has the same variance 
but lower variance based on linearity of expectation and variance. 

$$
\begin{aligned}
\mathbb{E}[\bar{X}_n] & = \mathbb{E}\left[ \frac{1}{n} \sum_{i=1}^n X_i \right] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[X_i] = \frac{1}{n} n \mu = \mu  \\
\mathrm{Var}[\bar{X}_n] & = \mathrm{Var}\left [ \frac{1}{n} \sum_{i=1}^n X_i \right] = \frac{1}{n^2} \sum_{i=1}^n \mathrm{Var}(X_i) = \frac{1}{n^2} n \sigma^2 = \frac{\sigma^2}{n} \tag{10}
\end{aligned}
$$

With the equation (10), we can use $k$ hash functions instead of one
as follows.

| Stream $\rarr$ | 13 | 25 | 19 | 25 | 19 | 19 | val$_i$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $h_1$| 0.51 | 0.26 | 0.79 | 0.26 | 0.79 | 0.79 | 0.26 |
| $h_2$ | 0.22 | 0.83 | 0.53 | 0.84 | 0.53 | 0.53 | 0.22 |
| $\dots$ |$\dots$ | $\dots$  $\dots$ |$\dots$ | $\dots$  |$\dots$ | $\dots$ | $\dots$ | 
| $h_k$ | 0.27 | 0.44 | 0.72 | 0.44 | 0.72 | 0.72 | 0.27 |

Now, for
improved accuracy, we just take the average of the $k$ minimums first, before reverse-solving. 

With the Chebyshev's inequality, we can show

$$
\mathrm{Pr}(|X - \mathbb{E}[\bar{X}]| \geq \epsilon \mathbb{E}[\bar{X}]]) \leq \frac{\mathrm{Var}[\bar{X}]}{(\epsilon \mathbb{E}[\bar{X}])^2}  < \frac{1}{k \epsilon^2}
$$

If we want the error with the probability at most $\delta$, we can set
the value of $k$ as 

$$
k = \frac{1}{\delta \epsilon^2} \tag{11}
$$


## Median trick

In equation (11), the space complexity for the hash function grows 20 times 
if we set our failure rate $\delta = 0.05$,

$$
k = \frac{1}{0.05 \times \epsilon^2} = \frac{20}{\epsilon^2}
$$

If we want to reduce our space complexity, we could use the "median-of-means" 
technique for error and success amplification, and analyze it using the Chernoff
bound (Thomas Vidick, CMS139 notes). 

Now, we set $t = O(\log1/\delta)$ and run $t$ trials each with failure probability
$\delta = 1/5$, which gives us the value of $k = 5/\epsilon^2$ (the number of 
hash functions we need). 

Now, let $\hat{d}_1, \hat{d}_2, \cdots, \hat{d}_t$ be the outcomes of the
$t$ trails, which is the estimated value of the number of distinct elements. Then,
we set 

$$
\hat{d} = \mathrm{median}(\hat{d}_1, \hat{d}_2, \cdots, \hat{d}_t)
$$

We know that for the median $\tilde{\mu}$ of a random variable $X$ are 
all points $x$ such that 

$$
\mathrm{Prob}(X \leq \tilde{\mu}) = \mathrm{Prob}(X \geq \tilde{\mu}) = \frac{1}{2} \tag{12}
$$

Assume the true value is $d$, and we bound our error with the following range

$$
[(1-4\epsilon)d, (1+4\epsilon)d]
$$

Then, if more than half of trials fill in the above range, then the equation (12)
tells us that the median for sure will fall in the above range. Now, let's 
assume that $2/3$ of trials fall in the range, then the median will fall in too.

Since the failure probability $\delta = 1/5$ , then we know for all $t$ trials,
each of them can fall in the the error bound with probability at least $4/5$ 
(that's the meaning and purpose of failure probability). Now, we want to 
estimate _the probability that the median $\hat{d}$ falling in the 
error bound_ we set up.  

Let random variable $X$ be  the number of trails of falling in 
$[(1-4\epsilon)d, (1+4\epsilon)d]$, then we can have 

$$
\mathrm{Pr}\left ( \hat{d} \notin [(1-4\epsilon)d, (1+4\epsilon)d] \notin \right) \leq \mathrm{Pr}\left ( X < \frac{2}{3} t \right) \tag{13}
$$

The equations (13) means that the probability of median $\hat{d}$  does not 
fall in the range is bounded by ${Pr}\left ( x < \frac{2}{3} t \right)$ for 
the obvious reason (definition of median). 

Since we set up $\delta = 1/5$, which means that each trail falls in the 
range with probability at leas $4/5$, therefore we could have 

$$
\begin{aligned}
\mathbb{E}[X] & = \mathbb{E} \left [ \sum_{i=0}^t I_i (\text{falling in the range}) \right] \\
& =  \sum_{i=0}^t \mathbb{E} [ I_i (\text{falling in the range}) ]\\ 
& = \sum_{i=0}^t 1 \cdot \mathrm{Pr}[I_i (\text{falling in the range}) ] \\ 
& = \frac{4}{5} t \tag{14}
\end{aligned}
$$

Now, substitute the equation (14) into the equation (13), we can have 

$$
\begin{aligned}
\mathrm{Pr}\left ( \hat{d} \notin [(1-4\epsilon)d, (1+4\epsilon)d] \notin \right) & \leq \mathrm{Pr}\left ( X < \frac{2}{3} t \right) \\
& = \mathrm{Pr}\left ( X < \frac{5}{6} \cdot \mathbb{E}[X] \right)  \\
& = \mathrm{Pr}\left ( X -   \mathbb{E}[X] < - \frac{1}{6} \cdot \mathbb{E}[X] \right)  \\
& = \mathrm{Pr}\left ( \mathbb{E}[X] - X >  \frac{1}{6} \cdot \mathbb{E}[X] \right) \\
& \leq  \mathrm{Pr}\left ( | X - \mathbb{E}[X] | \geq  \frac{1}{6} \cdot \mathbb{E}[X] \right)
\end{aligned}
$$

Now, apply the following Chernoff bound (there are variations of this bound, see
[wikipedia](https://en.wikipedia.org/wiki/Chernoff_bound){:target="_blank"})

$$
\mathrm{Pr}(|X - \mu| \geq \epsilon \mu ) \leq 2 \exp \left ( \frac{-\epsilon^2 \mu}{3} \right),
$$

we could have 

$$
\mathrm{Pr}\left ( | X - \mathbb{E}[X] | \geq  \frac{1}{6} \cdot \mathbb{E}[X] \right) \leq 
2 \exp  \left ( \frac{-(1/6)^2 \cdot (4/5)t}{3} \right) = O(e^{-ct}) \tag{15}
$$

Since we set $t = O(\log1/\delta)$, equation (15) shows the error is bounded by
$\delta$ for sure. 

In summary, for $t = O(\log1/\delta)$ trials, each using $k = 1/(\epsilon^2 \delta)$
hash functions. With $\delta = 1/5$, the total space (hash function we need to 
store) we will use is

$$
k = \frac{5t}{\epsilon^2} = O \left( \frac{\log(1/\delta)}{\epsilon^2} \right)
$$

This means the space complexity does not depend on the number of distinct 
elements $d$ or the number of items in the stream $n$ (both of these numbers are
typically very large). 


## Algorithms in practice 


Right now, our algorithm uses continuous valued fully random hash functions.
Howwever, this can’t be implemented. Therefore, we need to extend the analysis
into the discrete field. 

In practice, it is pretty amazing that we can actually reduce the space 
complexity down to $\log \log n$ bits (maybe not that amazing as we are only
trying to get zero-moment from one data pass). This algorithm is 
often referred to as "approximate counting", "probabilistic counting2,
or simply the "LogLog algorithm" (with HyperLogLog being the most recent 
version {% cite  flajolet2007hyperloglog %} ). 

Before we implementing the Flajolet-Martin algorithm, let's frame the counting as a random process. When we talk about 'counting', the first
thing comes to our mind is probably a sequence of numbers: $1, 2, 3, \cdots N$. However, any sequence could be taken as a counting process 
naturally:

- a, b, c, d, e, ...
- user576, user213, user398, user470, ... 
- $0101101010...$

Now, let's do some magic stuff: _every time a new element comes in, we flip the coin, if it was a head, then we mark it as $1$ and put it
into a vector of length 32; if it was a tail, then we mark it as $0$ and put it the vector_.  

<div class='figure'>
    <img src="/math/images/flajolet-martin.png"
         alt="distinct values"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Illustration of
        hash function as a machine of flipping coins. 
    </div>
</div>


In the paper by [Flajolet and Martin (1985)](#flajolet1985probabilistic), they
noticed that if the values hash function returns are uniformly distributed,
the pattern of having $k$ consecutive tails (0) appears with probability 
$2^{-k-1}$, which is aligned with our common sense. Assume we flip a coin
three times, what's the probability of having 3 heads? It is $1/8 = 2^{-3}$.

Now, let's run a simulation with `Python`. We want to run different times 
of simulation and calculate the probability of having 3 consecutive trailing zeros 
when we hash different amount of elements into the bit string of length 32. 


<div class='figure'>
    <img src="/math/images/fm_sim_plot.png"
         alt="distinct values"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> plot of simulations that
        calculates the probability of having 3 consecutive trailing zeros
        when we hash different amount of distinct elements. 
    </div>
</div>

Firgure 3 shows that the probability of having $3$ consecutive trailing
zeros becomes stable, which is around $0.06153$ when we have $1 \times 10^5$
distinct values hashed as a bitstring of length 32, which is close to

$$
\frac{1}{2^4} = 0.0625
$$

It is this pattern that gives Flajolet and Martin's idea of probabilistic 
counting algorithm, which means that the probability of a hash output 
ending with $2^k$ (a one followed by $k$ zeros, such as $...1000$) is 
$2^-(k+1)$, since this corresponds to flipping $k$ heads and then a tail
with a fair coin. 

Now, we will use this idea to count the distinct values. We assume that we have
a hash function $h$ that maps each element to a bitstring of length $s$ (32), i.e.

$$
h: [n] \to \{0, 1\}^s
$$

We assume that $h$ is truly random, which means that for each $i \in [n]$,
$h(i)$ is chosen uniformly at random from $\{0, 1\}^s$. 

Next, for each $i \in [n]$, let $r(i)$ be the number of trailing 0's in $h(i)$.
For example, 

$$
h(42) = 10111100010110001010010000110110, \quad \text{(1 zero)}
$$

so in this case, $r(42) = 1$. A useful observation about this quantity $r(i)$
is that: 

$$
\mathrm{Pr} [ r(i) \geq k] = \frac{1}{2^k} \tag{16}
$$

_Remark:_ equation (16) gives the cumulative probability whereas the Figure 3 only
shows the probability of $k= 3$. Here is the algorithm. 


```python
def fm_algorithm(num = 1000):
    """
    plot the probabilistic distinct elements counting algorithms
    source: https://en.wikipedia.org/wiki/Flajolet%E2%80%93Martin_algorithm
    """
    bitmap = '0'*32
    z = 0
    for i in range(num):
        hash_int = mmh3.hash(str(i), signed=False)
        hash_str = "{0:b}".format(hash_int)
        count_trailing0s = _calculate_trailing0s(hash_str)

        bitmap = bitmap[:count_trailing0s] + '1' + bitmap[count_trailing0s+1:]

    r = bitmap.find('0')
    
    return 2**r
```

As we have analyzed that using one hash function does not give very accurate
estimation. We need to deploy more hash functions and use median trick to 
improve the accuracy. 


<div class='figure'>
    <img src="/math/images/fm_count_plot.png"
         alt="distinct values"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> plot of counting distinct
        elements with Flajolet-Martin algorithm. 
    </div>
</div>


{% endkatexmm %}

