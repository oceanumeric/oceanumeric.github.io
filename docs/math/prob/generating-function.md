<style>
    [data-md-color-scheme="default"] {
    background-color:#fffff8;
    .md-content > .md-typeset {
        color: #16291c;
    }
    }
    .md-content {
    font-family: 'Latin Modern Roman', Palatino, Georgia, Cambria, 'Times New Roman', Times, serif;
    /* background-color:#fffff8; */
    }
    .md-content > .md-typeset {
    font-size: 13pt;
    font-style: normal;
    font-weight: normal;
    text-align:justify;
    }
</style>

# Generating Function 


In mathematics, a generating function is a way of encoding an infinite sequence
of numbers ($a_n$) by treating them as the coefficients of a formal power series.
It is a very powerful tool to study probability. 

According to George Pólya, a generating function is a device somewhat similar to a bag. Instead of 
carrying many little objects detachedly, which could be embarrassing, 
we put them all in a bag, and then we have only one object to carry, the bag.
Herbert Wilf states a generating function is a clothesline on which we hang up a sequence of numbers for display.

In this post, I will show you how generating function could help us to understand
probability more intuitively.

## Basic Concepts of Probability

A probability is a number that reflects the chance or likelihood that a 
particular event will occur. Probabilities can be expressed as proportions 
that range from 0 to 1, where, roughly speaking, 0 indicates impossibility of 
the event and 1 indicates certainty.

The concept of probability is built up on the concept of [countable set](https://en.wikipedia.org/wiki/Countable_set). 
For instance, the probability of rolling any given number for a standard dice is
$1/6$ as set $S = \{1, 2, 3, 4, 5, 6 \}$ is a countable set and the total number
of elements is 6. When the random variable is continuous, the concept of probability
relies on a [$\sigma$-algebra](https://en.wikipedia.org/wiki/%CE%A3-algebra) to
have a measurable space.

### Power set

In probability theory, an event is a set of outcomes of an experiment 
(a subset of the sample space) to which a probability is assigned. For instance,
an event could be $A= \{3\}$ when the sample space is $S=\{1, 2, 3, 4, 5\}$.
Assuming each event is independent and happens with the same chance, then we
could have:

$$P(A = \{3\}) = \frac{1}{5} = P(A = \{1\}) = P(A = \{2\})$$

How many subsets does a sample space, such as $S=\{1, 2, 3, 4, 5 \}$, have? To
calculate the number of all subsets, we could simply list them:

* null set $\emptyset$
* subsets with one element: $\{1\}, \{2\}, \{3\}, \cdots$
* subsets with two elements: $\{1, 2\}, \{1, 3\}, \cdots, \{2, 3\}, \cdots$
* subsets with three elements:
* subsets with four elements: 
* sample space itself: $S=\{1, 2, 3, 4, 5\}$

This way of calculating the [power set](https://en.wikipedia.org/wiki/Power_set)
is not efficient. We could design the following algorithm to calculate the 
total number of subsets:

- construct an indicator function $I_A: S \to \{0, 1\}$, and it indicates 
whether an element of $S$ belongs to a subset $A$ or not;
- then, we could have a binary number with $N=5$ bits such as $d = 00000$, where
each bit represents whether an element will be chosen or not
- for instance, a subset $s_2 = \{3, 5\}$ could be represented as $00105$. 
- therefore, the total number of subsets is $|S| = 2^N = 2^5$

Okay, now we know how to calculate the number of power set: $|S| = 2^N$. Let's 
turn to binomial theorem for a while and see how could connect binomial theory
with power set. This will lead us to a better understanding of generating 
function. 

### Binomial theorem

According to binomial theorem, it is possible to expand any nonnegative power 
of $x + y$ into a sum of the form

$$
\begin{align}
(x+y)^n & = \binom{n}{0}x^n y^0 +  \binom{n}{1} x^{n-1}y + \binom{n}{2} x^{n-2}y^2 + \cdots \nonumber \\
& \quad \quad \cdots + \binom{n}{n-1} x^1y^{n-1} + \binom{n}{n} x^0y^n  \nonumber \\
& = \sum_{k=0}^n \binom{n}{k}x^ky^{n-k}
\end{align}
$$ 

,where the coefficient is given by the formula
$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$
The [proof](https://en.wikipedia.org/wiki/Binomial_theorem) for this theorem 
is easy to follow. Let's go through an example together and see its connection
with power set. 

Let's expand the following equation:

$$
\begin{align*}
(x+y)^3 & = (x+y)(x+y)(x+y) \\
        & = xxx + xxy + \cdots + yyx + yyy \\
        & = x^3 + 3x^2y + 3xy^2 + y^3 
\end{align*}
$$

The coefficient of $xy^2$ is $3$ which equals $\binom{3}{2}$ because there are
three _events_ that $x$ was chose and two $y$s were chose: $\{xyy, yxy, yyx\}$,
corresponding to the three $2$-element subsets of $\{1, 2, 3\}$, namely,
$$\{2,3\}, \{1,3\}, \{1,2\}$$,
where each subset specifies the positions of the y in a corresponding string.

How many elements does the expansion of $(x+y)^3$ have? It has the same number
of elements with power set $|S|=2^3$. Why? To understand this, we could use
binary number again. 

For an expansion of $(x+y)^n$, we could map $n$ as the
number of bit, map $x$ as $0$, and map $y$ as $1$. Instead of seeing an expansion
as a process of multiplications, we could treat the expansion as a process of
_grouping (or combining)_ elements. 

See the following expansion,

$$
\begin{align*}
(x+y)^3 & = (x+y)(x+y)(x+y) \\
        & = xxx + xxy + xyx + xyy + yxx + yxy + yyx + yyy \\
        & = 000 + 001 + 010 + 011 + 100 + 101 + 110 + 111
\end{align*}
$$

At this stage, the connection between power set and binomial theorem should be
very clear. 

### Representing power set with polynomials 

Having a grip with the connection between power set and binomial theorem, one
could represent power set of a sample space with a polynomial. 

With a sample space $S=\{1, 2, 3, 4, 5 \}$, we could use the following
polynomial to represent its power set:

$$g(x) = (1+x^1)(1+x^2)(1+x^3)(1+x^4)(1+x^5)$$

The reason we use the above polynomial to represent power set is that polynomial
functions have been studied well and many properties are also well known. You
could also use the following equation to represent the power set

$$g(x) = (1+x_1)(1+x_2)(1+x_3)(1+x_4)(1+x_5)$$

In this case $x_i$ is just a dummy variable. If you are confused, see the
following expansion:

$$
\begin{align*}
g(x, y) & = (x+y)^5 \\
g(x, 1)  & = (1+x)^5 \\
g(x)    & = (1+x_1)(1+x_2)(1+x_3)(1+x_4)(1+x_5) 
\end{align*}
$$ 

??? quote "Question"
    Let the 𝑋={1,2,…,2000}. How many subsets 𝑇 of 𝑋 are such that the sum 
    of the elements of 𝑇 is divisible by 5?
    <br>
    To see how polynomials could be used to answer this question that might 
    look like a mission impossible, you can watch this
    <a href="https://youtu.be/bOXCLR3Wric">video</a> by legendary 3Blue1Brown.


## Introducing Generating Function

The general idea of generating function has much wider scope than its 
applications to probability, which means it as many applications in
mathematics as a technique. In this post, we will focus on its application
in _Discrete Math_. 


### Ordinary generating function

The ordinary generating function for the sequence 
$\left\langle a_{0}, a_{1}, a_{2}, a_{3} \ldots\right\rangle$ is the power series:

$$
G(x)=a_{0}+a_{1} x+a_{2} x^{2}+a_{3} x^{3}+\cdots .
$$

There are a few other kinds of generating functions in common use, but ordinary 
generating functions are enough to illustrate the power of the idea, 
so we'll stick to them and from now on, generating function will mean 
the ordinary kind (MIT, Math [6.042J-Chapter 12](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/resources/mit6_042jf10_chap12/)). 

A generating function is a "formal" power series in the sense that we usually 
regard $x$ as a placeholder rather than a number. Only in rare cases will 
we actually evaluate a generating function by letting $x$ take a real number 
value, so we generally ignore the issue of convergence (ibid). 

We could indicate the correspondence between a sequence and its 
generating function with a double-sided arrow as follows:

$$
\left\langle a_{0}, a_{1}, a_{2}, a_{3}, \ldots\right\rangle \longleftrightarrow a_{0}+a_{1} x+a_{2} x^{2}+a_{3} x^{3}+\cdots
$$

For example, here are some sequences and their generating functions:

$$
\begin{aligned}
&\langle 0,0,0,0, \ldots\rangle \longleftrightarrow 0+0 x+0 x^{2}+0 x^{3}+\cdots=0 \\
&\langle 1,0,0,0, \ldots\rangle \longleftrightarrow 1+0 x+0 x^{2}+0 x^{3}+\cdots=1 \\
&\langle 3,2,1,0, \ldots\rangle \longleftrightarrow 3+2 x+1 x^{2}+0 x^{3}+\cdots=3+2 x+x^{2}
\end{aligned}
$$

Again, the idea of generating function is very similar to the connection 
between power set and binomial theorem. The takeaway is that we can use
polynomials to represent a series of combinations (or a subset of a sample space
in probability).

### Probability generating functions

Now, we will formally define probability generating function and study its 
properties and applications. 

we know for a sequence $\left\{a_{i}: i=0,1,2, \ldots\right\}$ of real numbers, 
the ordinary generating function of the sequence is defined as

$$
G(s)=\sum_{i=0}^{\infty} a_{i} s^{i}
$$

for those values of the parameter $s$ for which the sum converges. For a given 
sequence, there exists a radius of convergence $R(\geq 0)$ such that the 
sum converges absolutely if $|s|<R$ and diverges if $|s|>R$. $G(s)$ may be 
differentiated or integrated term by term any number of times when $|s|<R$. The
probability generating function is just one kind of ordinary generating function.
 
<div class="definition">
Let $X$ be a discrete random variable taking values in the non-negative integers 
$\{0, 1, 2, \cdots \}$. The probability generating function (PGF) of $X$ is
defined as 

$$
G_{X}(s)=\mathrm{E}\left(s^{X}\right) = \sum_{k=0}^{\infty} p_{k} s^{k},
$$

where $p_k = P(X = K), k = 0, 1, \cdots$ is the probability of $X = k$.
Note that $G_{X}(1)=1$, so the series converges absolutely for $|s| \leq 1 .$ Also $G_{X}(0)=p_{0}$.
</div>
<span> </span>


### Power series

Before we deriving properties of probability generating function, let's review
power series first. 

We know, for a geometric series, we have the formula,

$$
\begin{aligned}
1 + r + r^2 + r^3 + \cdots = \sum_{n=0}^{\infty} r^n = \frac{1}{1-r}, \quad \quad \text{when $|r|<1$.}
\end{aligned}
$$

With this formula, we could calculate probability generating function for different
discrete random variables knowing their distribution. Notice that value of
probability is always less 1, which guarantees the convergence of $G_{X}(s)$. 

### Properties of probability generating function 

It is easy to derive properties of probability generating function as follows:

* $G_{X}(0) = P(X = 0)$
* $G_{X}(1) = 1$

Just substitute values of $s$ into the formula in definition 1, we could have:

$$
\begin{aligned}
G_X(0) & = 0^0 \times P(X=0) + 0^1 \times P(X=1) + 0^2 \times P(X=2) + \cdots \\
       & = P(X=0) \\
G_X(1) & = \sum_{k=0}^{\infty} 1^k P(X=k) = \sum_{k=0}^{\infty} P(X=k) = 1 
\end{aligned}
$$


## Applications of PGF 


There are many applications of probability generating function(PGF), we will just
present some simple examples. First, let's use binomial distribution to
calculate its PGF.

__Example__ 2. Let $X \sim \text{Binomial}(n, p)$, so $P(X=k) = \binom{n}{k}p^kq^{n-k}$ 
for $x = 0, 1, \cdots, n$, where $q = 1-p$. Then, we could have

$$
\begin{aligned}
G_X(s) & = \sum_{k=0}^n s^k \binom{n}{k}p^kq^{n-k} \\
       & = \sum_{k=0}^n \binom{n}{k}(ps)^kq^{n-k} \\
       & = (ps + q)^n \quad \quad \quad \text{see equation(1)}\\
\end{aligned}
$$

based on the equation 1 in [binominal theorem](#binomial-theorem). 


__Example__ 3. Let $X \sim \operatorname{Poisson}(\lambda)$, so $P(X=x)=\frac{\lambda^{x}}{x !} e^{-\lambda}$ for $x=0,1,2, \ldots$

$$
\begin{aligned}
G_{X}(s)=\sum_{x=0}^{\infty} s^{x} \frac{\lambda^{x}}{x !} e^{-\lambda} &=e^{-\lambda} \sum_{x=0}^{\infty} \frac{(\lambda s)^{x}}{x !} \\
&=e^{-\lambda} e^{(\lambda s)} \quad \quad \quad \text { for all } s \in \mathbb{R}
\end{aligned}
$$

Thus $G_{X}(s)=e^{\lambda(s-1)} \quad$ for all $s \in \mathbb{R}$.

### Using PGF to calculate probability 

The probability generating function gets its name because the power series can be expanded and differentiated to reveal the individual probabilities. Thus, given only the PGF 
$G_{X}(s)=E \left(s^{X}\right)$, we can recover all probabilities $P(X=x)$.
For shorthand, write $p_{x}=P(X=x)$. Then

$$
G_{X}(s)=E \left(s^{X}\right)=\sum_{x=0}^{\infty} p_{x} s^{x}=p_{0}+p_{1} s+p_{2} s^{2}+p_{3} s^{3}+p_{4} s^{4}+\ldots
$$

Thus $p_{0}=P(X=0)=G_{X}(0)$.

Its first derivative:

$$G_{X}^{\prime}(s)=p_{1}+2 p_{2} s+3 p_{3} s^{2}+4 p_{4} s^{3}+\ldots$$

Thus $p_{1}=P(X=1)=G_{X}^{\prime}(0)$.

Its second derivative:

$$G_{X}^{\prime \prime}(s)=2 p_{2}+(3 \times 2) p_{3} s+(4 \times 3) p_{4} s^{2}+\ldots$$

Thus $p_{2}=P(X=2)=\frac{1}{2} G_{X}^{\prime \prime}(0)$.

Its third derivative: 

$$G_{X}^{\prime \prime \prime}(s)=(3 \times 2 \times 1) p_{3}+(4 \times 3 \times 2) p_{4} s+\ldots$$

Thus $p_{3}=P(X=3)=\frac{1}{3 !} G_{X}^{\prime \prime \prime}(0)$.

### Using PGF to calculate moments 

As well as calculating probabilities, we can also use the PGF to calculate the 
moments of the distribution of $X$. The moments of a distribution are 
the mean, variance, etc.

__Theorem__  _Let $X$ be a discrete random variable with PGF $G_{X}(s)$. Then_:

1. $E(X)=G_{X}^{\prime}(1)$
2. $E\{X(X-1)(X-2) \ldots(X-k+1)\}=G_{X}^{(k)}(1)=\left.\frac{d^{k} G_{X}(s)}{d s^{k}}\right|_{s=1}$ (This is the $k$ th factorial moment of $X$).



_Proof._

$$
\begin{aligned}
G_{X}(s) & =\sum_{x=0}^{\infty} s^{x} p_{x} \\
G_{X}^{\prime}(s) & =\sum_{x=0}^{\infty} x s^{x-1} p_{x} \\ 
\Rightarrow G_{X}^{\prime}(1)& =\sum_{x=0}^{\infty} x p_{x}=E(X)
\end{aligned}
$$

$$
\begin{aligned}
G_{X}^{(k)}(s)=\frac{d^{k} G_{X}(s)}{d s^{k}}& =\sum_{x=k}^{\infty} x(x-1)(x-2) \ldots(x-k+1) s^{x-k} p_{x}\\
 G_{X}^{(k)}(1)& =\sum_{x=k}^{\infty} x(x-1)(x-2) \ldots(x-k+1) p_{x}\\
&=E \{X(X-1)(X-2) \ldots(X-k+1)\}
\end{aligned}
$$

<span style="float:right">&#x25FB</span>
<br>

__Example__ 3. Let $X \sim \text{Poisson}(\lambda).$ The PGF of $X$ is $G_X(s) = e^{\lambda(s-1)}$.
Find $E(X)$ and Var($X$). 

We know that 

$$
\begin{aligned}
G'_X(s) & = \lambda e^{\lambda(s-1)} \\
\Rightarrow  E(X) & = G'_X(1) = \lambda  \\
\\ 
E\{X(X-1)\} & = G''_X(1) = \lambda^2e^{\lambda(s-1)}|_{s=1} = \lambda^2 \\
Var(X) & = E(X^2) - (E[X])^2 \\
       & = E\{X(X-1)\} + E(X) - (E[X])^2 \\
       & = \lambda^2 + \lambda - \lambda^2 \\
       & = \lambda 
\end{aligned}
$$ 

<strong>References</strong>
<br>
[1] [Rachel Fewster](https://www.stat.auckland.ac.nz/~fewster/325/notes/ch4.pdf)
<br>
[2] [Gleb Gribakin](http://www.am.qub.ac.uk/users/g.gribakin/sor/Chap3.pdf)
<br>
[3] [MIT OCW](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/aadb9105d1139c866272ca0425fc6440_MIT6_042JF10_chap12.pdf)