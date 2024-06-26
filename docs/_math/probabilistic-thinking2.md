---
title: Develop Some Fluency in Probabilistic Thinking (Part II)
subtitle: The foundation of machine learning and data science is probability theory. In this post, we will develop some fluency in probabilistic thinking with different examples, which prepare data scientists well for the sexist job of the 21st century. 
layout: math_page_template
date: 2023-02-16
keywords: probabilistic-thinking mathematics algorithms Concentration Bounds Markov's inequality Chebyshev's inequality approximate counting Robert Morris hash table 
published: true
tags: probability algorithm data-science machine-learning bernoulli-trial hypergeometric-distribution binomial-distribution 
---

This the second part of this series of posts as the title has stated. We
will continue to study the material from Cameron Musco's course - [Algorithms for Data Science](https://people.cs.umass.edu/~cmusco/CS514F22/index.html){:target="_blank"} {% cite musco2022 %}. This section assumes readers know what hash table is. If not, just check out 
[this video](https://youtu.be/Nu8YGneFCWE){:target="_blank"} from MIT. 

- [Example 5: Randomized load balancing](#example-5-randomized-load-balancing)
- [Example 6: flipping coins](#example-6-flipping-coins)
- [Example 7: Hoeffding’s inequality](#example-7-hoeffdings-inequality)
- [Example 8: Bernstein inequality](#example-8-bernstein-inequality)



{% katexmm %}

When we use a random hash function $h(x)$ to map keys into hashing numbers,
we could design a fully random one which has the following property:

$$
\mathbb{P}(h(x) = i) = \frac{1}{n}
$$ 



However, wit the above property, we need to store a table of $x$ values and
their hash values. This would take at least $O(m)$ space and $O(m)$ query time
to look up, making our whole quest for $O(1)$ query time pointless. Therefore
we need to find another property that will avoid collision and $O(1)$ query
time. 


__2-Universial Hash Function__ (low collision probability). A random hash 
function from $h: U \to [n]$ is two universal if :

$$
\mathbb{P}[h(x) = h(y)] \leq \frac{1}{n} \tag{1}
$$

One of the functions that are often used is the following one: 

$$
h(x) = (a x + b \  \mathrm{mod} p) \  \mathrm{mod} \  n 
$$ 



__Pairwise Independent Hash Function.__ A random hash function from
$h: U \to [n]$ is pairwise independence if for all $i, j \in [n]$:

$$
\mathbb{P}[h(x) = i \cap h(y) = j] = \frac{1}{n^2} \tag{2}
$$

We can show that pairwise hash function are 2-universal. Now, for $i \in [n]$,
it has $n-1$ pairs and one pair with itself, therefore:

$$
\begin{aligned}
\mathbb{P}[h(x) = h(y)] & = \sum_{j=1}^n \mathbb{P}[h(x) = i \cap h(y) = j] \\
& = n \cdot \frac{1}{n^2} = \frac{1}{n}
\end{aligned}
$$


## Example 5: Randomized load balancing

We have $n$ requests _randomly_ assigned to $k$ servers. How many requests 
must each server handle? Let's assume $R_i$ as the number of requests 
assigned to server $i$. The expectation tells 

$$
\begin{aligned}
\mathbb{E}[R_i] & = \sum_{j=1}^n \mathbb{E} [ \mathbb{I}_{\text{request j assigned to i}}] \\
& = \sum_{j=1}^n \mathbb{P}(\text{j assigned to i}) \times 1  \\
& = \frac{n}{k} \tag{3}
\end{aligned}
$$

If we provision each server be able to handle twice  the expected load,
what is the probability that a server is overloaded? We can calculate it 
based on Markov's inequality:

$$
\mathbb{P}[R_j  > 2 \mathbb{E}[R_i]] \leq \frac{\mathbb{E}[R_i]}{2\mathbb{E}[R_i]} = \frac{1}{2}
$$


### Chebyshev's inequality 

With a very simple twist, Markov’s inequality can be made
much more powerful. For any random variable $X$ and any value $t > 0$, we have: 

$$
\mathbb{P}(|X| \geq t ) = \mathbb{P}(X^2 \geq t^2 ) \leq \frac{\mathbb{E}[X^2]}{t^2}
$$

Now, by plugging in the random variable $X - \mathbb{E}$, we can hae 
Chebyshev's inequality.


__Chebyshev's inequality.__ For any random variable $X$ and $t > 0$,

$$
 \mathbb{P}(|X - E[X]| \geq t ) \leq \frac{\mathrm{Var}[X]}{t^2} \tag{4}
$$


With Chebyshev's inequality, we can calculate the probability that $X$
falls $s$ standard deviations from its mean as:

$$
\mathbb{P}(|X - E[X]| \geq s \cdot \sqrt{\mathrm{Var}[X]} ) \leq \frac{\mathrm{Var}[X]}{s^2\mathrm{Var}[X]} = \frac{1}{s^2} \tag{5}
$$

Equation (5) gives us the quick estimation on deviation of our estimations. It 
could also shows  that the sample average will always concentrate on 
mean with enough samples $n$ (law of large numbers). 

Consider drawing independent identically distributed (i.i.d) random variables
$X_1, \cdots, X_n$ with mean $\mu$ and variance $\sigma^2$. How well 
does the sample average 

$$ S = \frac{1}{n} \sum_{i=1}^n X_i$$ 

approximate the true mean $\mu$?

First, we need to calculate the sample variance

$$
\mathrm{Var}[S] = \mathrm{Var} \left [ \frac{1}{n} \sum_{i=1}^n X_i \right] = \frac{1}{n^2} \sum_{i=1}^n \mathrm{Var} [X_i] = \frac{1}{n^2} \cdot n \cdot \sigma^2 = \frac{\sigma^2}{n}
$$

By Chebyshev's inequality, for any fixed value $\epsilon > 0$, we have 

$$ 
\mathbb{P}( |S- \mu| \geq \epsilon) \leq \frac{\mathrm{Var}[S]}{\epsilon^2} = \frac{\sigma^2}{n \epsilon^2} \tag{5}
$$

Equation (6) could be used to estimate the size of sample we need for a certain
level of accuracy of the estimation. Instead of using $\epsilon$, we will use 
$\epsilon \cdot \sigma$, then we can have

$$ 
\mathbb{P}( |S- \mu| \geq \epsilon \cdot \sigma) \leq \frac{\mathrm{Var}[S]}{\epsilon^2 \sigma^2} = \frac{1}{n \epsilon^2} \tag{6}
$$

We can do quick estimation on sample size we need to achieve a certain level 
of estimation based on equation (6). Figure 1 shows that the probability 
of our sample mean deviating 10% percent of its variance is less or equal to
$0.1001$ when the sample size is greater than $100$, whereas the probability
is still bigger than 1 if we want our estimation with high accuracy ($\sigma = 0.01$),
which means that one needs a larger sample for that. 


<div class='figure'>
    <img src="/math/images/estimation_sample_size.png"
         alt="A demo figure"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Plot of probability against
        sample size based on equation (6).  <i>Remark</i>: the probability 
        is bounded by 1, the bound value calculation does not make too much 
        sense for dotted line. 
    </div>
</div>


Now, back to our balancing load example, we have assumed that each server
could _be able to handle twice the expected load_ and the expected load 
is 

$$\mathbb{E}[R_j] = \frac{n}{k}$$ 

as it has been derived in equation (3) and we want to find the probability that a server is overloaded. We can use 
Chebyshev's inequality to calculate it. However, we need to calculate the 
variance first. Let's denote $R_{i, j}$ as $1$ if $j$ is assigned to 
server $i$ and $0$ otherwise, then 

$$
\begin{aligned}
\mathrm{Var}[R_{i,j}] & = \mathbb{E} \left [ (R_{i, j} - \mathbb{E}[R_{i,j}])^2 \right ] \\
& = \mathbb{P}(R_{i, j}= 1) \cdot (1 - \mathbb{E}[R_{i, j}])^2 + \mathbb{P}(R_{i,j} = 0)(0 - \mathbb{E}[R_{i, j}])^2 \\
& =\frac{1}{k} \cdot\left(1-\frac{1}{k}\right)^2+\left(1-\frac{1}{k}\right) \cdot\left(0-\frac{1}{k}\right)^2 \\
& =\frac{1}{k}-\frac{1}{k^2}  \\
& \leq \frac{1}{k}
\tag{7}
\end{aligned} 
$$

Now, with the _linearity of variance_, we can have 

$$ \mathrm{Var}[R_i] = \sum_{j=1}^n \mathrm{Var}[R_{i, j}]  \leq \frac{n}{k} $$ 


Now, with $\mathbb{E}[R_j] = \frac{n}{k}$ and $ \mathrm{Var}[R_i]  \leq \frac{n}{k}$,
we can apply Chebychev's inequality ($R_i$ is the number of requests sent to
server $i$)

$$
\begin{aligned}
\mathbb{P}(R_i \geq \frac{2n}{k})  & = \mathbb{P}(R_i - \frac{n}{k} \geq \frac{n}{k})  \\
& \leq  \mathbb{P}( |R_i - \frac{n}{k}| \geq \frac{n}{k}) \quad \text{(take absolute value)} \\
&  = \mathbb{P}( |R_i - | \mathbb{E}[R_j] \geq \frac{n}{k}) \\
& \leq \frac{ \mathrm{Var}[R_{i, j}]}{(n/k)^2} \\ 
& \leq \frac{k}{n} \tag{8}
\end{aligned}
$$

Equation (8) shows that overload probability is extremely small when
$k << n$,  which seems counterintuitive as it means bound gets worse as
$k$ grows (more servers do not help). When $k$ is large, the number of
requests each server sees in expectation is very small so the law of 
larger numbers does not 'kick in'. 

### Maximum server load 

We have shown the probability of one server overloaded in equation (8), now
we are interested in the probability of maximum server load exceeds its
capacity. To get the maximum server load, we need to have a list of servers 
that overloaded, this means not just one sever but possible many of them are
overloaded. The value of maximum load on any server can be written as the random variable:

$$S = \max_{i \in [1, \cdots, k]} R_i$$ 

How could we form a probability for $S \geq (2n)/k$ in terms of $R_i$?
Let's start from null case, which is that no $R_i$ is greater or equal to 
$(2n)/k$. For this kind of case, $S < (2n)/k$. Therefore, we need to have
at least one of $R_i$ is greater or equal to $(2n)/k$. On the other side,
if all of $R_i$ is greater or equal to $(2n)/k$, then there must be the case
that $S \geq (2n)k$. However, we do not need all of them. We can transform them
as the 'or' problem that guarantees the event $S$:

$$
\begin{aligned}
\mathbb{P}(S \geq \frac{2n}{k})  & = \mathbb{P} \left ( \left[ R_1 \geq \frac{2n}{k} \right ] \ \text{or} \ \left[ R_2 \geq \frac{2n}{k} \right ] \ \text{or} \ \cdots \ \text{or} \ \left[ R_k \geq \frac{2n}{k} \right ] \right ) \\
& = \mathbb{P} \left ( \cup_{i=1}^k \left[ R_i \geq \frac{2n}{k} \right ] \right )
\end{aligned}
$$

__Union Bound.__ For any random events $A_1, A_2, \cdots, A_k$, we have

$$
\mathbb{P}(A_1 \cup A_2 \cup \cdots \cup A_k) \leq \mathbb{P}(A_1) + \mathbb{P}(A_2) + \cdots + \mathbb{P}(A_k) \tag{9}
$$

With the union bound, it is easy to show 

$$
\begin{aligned}
\mathbb{P}(S \geq \frac{2n}{k})  & = \mathbb{P} \left ( \cup_{i=1}^k \left[ R_i \geq \frac{2n}{k} \right ] \right )  \\ 
& \leq \sum_{i=1}^k \mathbb{P}  \left ( \left[ R_i \geq \frac{2n}{k} \right ] \right ) \\ 
& \leq \sum_{i=1}^k \frac{k}{n} \\
& \leq \frac{k^2}{n}
\end{aligned}  
$$

This means as long as $k \leq O(\sqrt{n})$, with good probability, the 
maximum server load will be small (compared to the expected to load). 


## Example 6: flipping coins

You might think this example will be super easy as we learned this example
probably from our elementary school. However, if I tell you that we flip
$n =100$ independent coins, each are heads with probability $1/2$ and
tails with probability $1/2$. Let $H$ be the number of heads, then we 
could have:

$$
\mathbb{E}[H] = \frac{n}{2} = 50; \quad \mathrm{Var}[H] = \frac{n}{4} = 25 \tag{10}
$$

When you look at the equation (10), it seems very straightforward. However,
to get numbers like $50, 25$, we need a long derivation. Let's use the definition
of expectation. _I have to admit that I found it very unnatural to write 
down the formula of expectation for our example_. I had an almost irresistible
impulse to review the definition of expectation again. 

<p class='theorembox'>
<b>Expected value</b>
<br>
Informally, the expected value is the weighted arithmetic mean of a large number of independently selected outcomes of a random variable. This means we have:
<i>one random variable</i> and <i> a countable set of possible outcomes</i>. 
For example, weight of human being follows the normal distribution: random 
variable is weight of human being, a countable set of possible outcomes 
is the set of all possible values of weight (some people are fit, some are
slim, some are overweight). 

At last, we need to calculate the <b>weighted</b> arithmetic mean. 
</p>

For the example of flipping coins, the random variable is $H$ (the number
of heads), the countable set of possible outcomes is the set of all
possible values $h \in [0, n]$. We know each trial follows the binomial
distribution, then we can write down the formula for the expected value.

$$
\mathbb{E}[H] = \sum_{h=0}^n h \cdot  \mathbb{P}(h) = \sum_{h=0}^n h \cdot \binom{n}{h}p^h(1-p)^{n-h} \tag{11}
$$

I hope you feel released when you compare equation (11) and (10), which shows
that it is natural for anyone to have unnatural reactions at the beginning. Now,
we will derive the expected value. There are different ways to do it (see
[all proofs](https://proofwiki.org/wiki/Expectation_of_Binomial_Distribution){:target="_blank"}). With binomial theory, it can be shown that 

$$ \mathbb{E}[H] = np$$


we will leverage the linearity of expectation again 
because each trial is independent. We can treat $H$ as the sum of 
discrete random variable $h$ that follows the Bernoulli trial, this means

$$
H = \sum_{i=1}^n h_i \tag{12}
$$

_Remark_: Be careful about the index. In equation (11), $h$ is not a 
random variable, it is just a mathematical symbol for us to count the number of
heads (which could be $0$). However, in equation (12), $h_i$ is a random 
variable which follows the Bernoulli trials. We run $n$ independent trials
and we are interested in the sum of those discrete random varaibles. 

Therefore, 
$$
\begin{aligned}
\mathbb{E}[H] & = \mathbb{E} \left [ \sum_{i=1}^n h_i \right ] = \sum_{i=1}^n \mathbb{E}[h_i] = np  \tag{13}
\end{aligned}
$$


Bernoulli random variables and indicator variables are two aspects of the same concept. As a review, a random variable $I$ is called an indicator variable for an event $A$ if $I=1$ when $A$ occurs and $I=0$ if $A$ does not occur. $P(I=1)=P(A)$ and $E[I]=P(A)$. Indicator random variables are Bernoulli random variables, with $p=P(A)$. For instance, using indicator function, the expectation could
be calculated as follows:

$$
\mathbb{E}[H] = \sum_{i=1}^n \left [ \mathbb{I}_{\text{$h_i$=head}} \cdot \mathbb{P} \right]
$$

To use the same trick we did in equation (13), we could derive the variance
of $H$ is

$$\mathrm{Var}[H] = \mathrm{Var}  \left [ \sum_{i=1}^n h_i \right ] = \sum_{h=1}^n  \mathrm{Var}[h_i]  = np(1-p) \tag{14}
$$


Now, with the results in equation (11), we can test how concentrated of
Markov's inequality and Chebyshev's inequality:

$$
\begin{aligned}
& \mathbb{P}(X \geq a) \leq \frac{\mathbb{E}[X]}{a}  \\
& \mathbb{P}(|X - E[X]| \geq a) \leq \frac{\mathrm{Var}[X]}{a^2}
\end{aligned}
$$

<div class='figure'>
    <img src="/math/images/inequality_bounds.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Plot of different inequality
        bounds, which shows that the concentration improves a lot from
        Markov's inequality to Chebyshev's inequality; but there is still
        some space to improve. 
    </div>
</div>


## Example 7: Hoeffding’s inequality


Suppose that $Z$ has a finite mean and that $\mathbb{P}(Z \geq 0)=1$. Then, for any $\epsilon>0$,
$$
\mathbb{E}(Z)=\int_0^{\infty} z d P(z) \geq \int_\epsilon^{\infty} z d P(z) \geq \epsilon \int_\epsilon^{\infty} d P(z)=\epsilon \mathbb{P}(Z>\epsilon)
\tag{15}
$$
which yields Markov's inequality:
$$
\mathbb{P}(Z>\epsilon) \leq \frac{\mathbb{E}(Z)}{\epsilon}
$$

Equation 15 is very powerful as it implies that any __monotonic transformation__ preserve this property, which gives us other equalities, such
as Chernoff's bound and Bernstein inequality. 


Figure 2 shows that although Chebyshev's inequality is closer to the true 
probability value, however it does not decay very fast. Let's check the formula

$$
 \mathbb{P}(|X - E[X]| \geq \epsilon ) \leq \frac{\mathrm{Var}[X]}{\epsilon^2}
$$

We know $\mathrm{Var}[X_i] = \sigma^2/n$ if $X_i$ are iid with mean $\mu$ and 
variance $\sigma^2$, this means we have

$$
 \mathbb{P}(|X_i - \mu| \geq \epsilon ) \leq \frac{\sigma^2}{n \epsilon^2}
$$

While this inequality is useful, it does not decay exponentially fast as $n$ increases. To improve the inequality, we use Chernoff ’s method: for any 
$t > 0$, 

$$
\mathbb{P}(Z > \epsilon) = \mathbb{P}(e^z > e^{\epsilon})  = \mathbb{P}(e^{tz} > e^{t \epsilon}) \leq \frac{\mathbb{E}(e^{tz})}{ e^{t \epsilon}} \tag{16}
$$

This gives the inequality with the moment generating function. 


The moment-generating function is so named because it can be used to find the moments of the distribution. The series expansion of $e^{t X}$ is
$$
e^{t X}=1+t X+\frac{t^2 X^2}{2 !}+\frac{t^3 X^3}{3 !}+\cdots+\frac{t^n X^n}{n !}+\cdots .
$$
Hence
$$
\begin{aligned}
M_X(t)=\mathrm{E}\left(e^{t X}\right) & =1+t \mathrm{E}(X)+\frac{t^2 \mathrm{E}\left(X^2\right)}{2 !}+\frac{t^3 \mathrm{E}\left(X^3\right)}{3 !}+\cdots+\frac{t^n \mathrm{E}\left(X^n\right)}{n !}+\cdots \\
& =1+t m_1+\frac{t^2 m_2}{2 !}+\frac{t^3 m_3}{3 !}+\cdots+\frac{t^n m_n}{n !}+\cdots
\end{aligned}
$$

where $m_n$ is the $n$th moment. Differentiating $M_X(t) i$ times with respect to $t$ and setting $t=0$, we obtain the $i$th moment about the origin, $m_i$. 


To learn more about probability concentration, please read this [notes](../../pdf/Concentration.pdf){:target="_blank"}.



Before we present Hoeffding's inequality, we will
learn a new concept called symmetric random variable. 

A random variable $Z$ is _symmetric_ if $Z$ and $-Z$ have the same
distribution. A simple example of a symmetric random variable
is the well know _symmetric Bernoulli_, which takes values $-1$ and
$+1$ with equal probability $1/2$ each, i.e, 

$$\mathbb{P}[Z = 1] = \mathbb{P}[Z = -1] = 1/2$$ 

For a usual Bernoulli random variable ($X$) with two possible values
$1, 0$ and parameter $1/2$, we can transfer it into a symmetric
Bernoulli random variable by setting

$$Z = 2X - 1$$ 

__Hoeffding’s Inequality__ Let $X_1, \cdots, X_N$ be independent Bernoulli
random variables, and let $a = (a_1, \cdots, a_N) \in \mathbb{R}^N$. Then,
for any $t \geq 0$, we have 

$$
\mathbb{P} \left [\sum_{i=1}^N a_i X_i \geq t \right ] \leq  \exp \left ( - \frac{t^2}{2||a||_2^2} \right ) \tag{17}
$$

_Remark:_ sometimes, you see the above formula for two-side inequality with
absolute symbol. For the proof, you can find it [here](https://gclinderman.github.io/blog/probability/2018/01/07/concentration-inequalities.html){:target="_blank"} or
in the book by [Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html){:target="_blank"}. 

Now, to use this inequality, we have to be careful as our random variable 
is $X_i$ - each trial of flipping the coin, instead of $X$ in equation (10)
and figure 2. 

Let $X_i$ be the random variable of mapping head or tail of flipping the coin
into $1, 0$, we are interested in the value of the number of heads in $N = 100$ trials. To use equation (17), we need to transform our random variable
into symmetric one by setting 

$$Y_i = 2(X_i - \frac{1}{2})$$ 

Then we can have 

$$
\begin{aligned}
 \mathbb{P} \left [ \sum_{i=1}^N a_i X_i  > t \right ] &= \mathbb{P} \left[ \sum_{i=1}^N a_i \left(\frac{Y_i}{2} + \frac{1}{2}\right) > t \right ] \\
&= \mathbb{P} \left[ \sum_{i=1}^N Y_i>2t-N \right ] \quad a_i = 1 \ \forall i \in N \\ 
& \leq   \exp\left( - \frac{(2t-N)^2}{2 N}\right) \quad ||a||_2^2 = N 
\end{aligned}
$$


<div class='figure'>
    <img src="/math/images/inequality_bounds2.png"
         alt="Inequality bounds compare"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> Plot of different inequality
        bounds, which shows that Hoeffding's inequality follows the 
        cumulative density function very closely. 
    </div>
</div>




## Example 8: Bernstein inequality

__Bernstein Inequality__ Consider symmetric independent random
variables $X_1, \cdots, X_n$ all falling in $[-a, a]$ and $E[X_i] = 0$ (if not we can standardized it) with $E[X_i^2] = \sigma^2$. Then 
Let 


$$
\mathbb{P} \left [ \left | \sum_{i=1}^N X_i \geq t \right | \right] \leq 2 \mathrm{exp} \left ( - \frac{ t^2 }{2 n \sigma^2 + \frac{2}{3} a \cdot t } \right) \tag{18}
$$

Now, to apply the equation (18) in our flipping coins example, we could have

$$
\begin{aligned}
\mathbb{P} \left[ \sum_{i=1}^N X_i \geq t \right] & = \mathbb{P} \left[  \sum_{i=1}^N \left ( \frac{Y_i}{2} + \frac{1}{2}  \right ) \geq t  \right] \\
& = \mathbb{P} \left[  \sum_{i=1}^N  Y_i  \geq (2t-N) \right] \quad (\mu = \mathbb{E}[Y_i] = 0) \\
& \leq  \exp \left ( - \frac{ (2t-N)^2}{2 N  + \frac{2}{3} \cdot (2t -N)}  \right) \quad (a = 1; \sigma^2 = 1)
\end{aligned}
$$

_Remark:_ we drop $2$ because we are dealing with one side inequality. 

<div class='figure'>
    <img src="/math/images/inequality_bounds3.png"
         alt="Inequality bounds compare"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Plot of different inequality
        bounds, which shows that Hoeffding's inequality follows the 
        cumulative density function very closely and Bernstein's inequality gives the larger value  than Hoeffding's inequality in this case.
    </div>
</div>

To learn when Hoeffding is stronger than Bernstein and when Bernstein is 
stronger than Hoeffding, please read this [post](https://dustingmixon.wordpress.com/2014/09/29/an-intuition-for-concentration-inequalities/){:target="_blank"}.



__Summary__ Those inequalities provide the theoretical basis for many statistical machine learn-
ing methods. To learn more about related theories, please visit those sites:

- [Topics In Mathematics Of Data Science](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/){:target="_blank"}
- [High-Dimensional Probability and Statistics](https://people.math.wisc.edu/~roch/hdps/){:target="_blank"}
- [Mathematical Methods in Data Science](https://people.math.wisc.edu/~roch/mmids/){:target="_blank"}
- [The Modern Algorithmic Toolbox](https://web.stanford.edu/class/cs168/){:target="_blank"}






{% endkatexmm %}