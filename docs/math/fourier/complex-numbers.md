---
title: Complex Numbers
---

<style>
    [data-md-color-scheme="default"] {
    background-color:#fffff8;
    } 
    .md-content {
    font-family: TeX Gyre Schola;
    } 
    .md-content > .md-typeset {
    font-size: 13pt;
    font-style: normal;
    font-weight: normal;
    text-align:justify;
    }
    ol {
    counter-reset: list;
    }
    ol > li {
       list-style: none;
       position: relative;
       }
    ol > li:before {
       counter-increment: list;
       content: "(" counter(list, decimal) ") ";
       position: absolute;
       left: -1.7em;
       }
</style>


## Concepts and Properties

The set of complex numbers with addition and multiplication is a _field_ with
additive and multiplicative identities (0,0) and (1,0) as 

1. both $+$ and $\cdot$ are associative 
2. both $+$ and $\cdot$ are commutative 
3. the distributive law holds
4. both additive and multiplicative identities exist and unique 
5. both additive and multiplicative inverse exist and unique

__Definition 1.__ Let $a+bi$ be a complex number. The _complex conjugate_, or simply _conjugate_,
of $a+bi$ is defined as

$$\overline{a+bi} = a - bi$$

The _norm (or modulus)_ of $a+bi$ is defined to be 

$$|a+bi| = \sqrt{a^2+b^2}$$


For $z, w \in C$, we have:

* $\overline{z+w} = \overline{z} + \overline{w}$ 
* $\overline{z\cdot w} = \overline{z} \cdot \overline{w}$ 
* $|zw| = |z| |w|$ 
* $z \overline{z} = a^2 + b^2 = |z|^2$


__Definition 2.__ Let $X$ be a nonempty set. A _metric_ on $X$ is a function
$d: X \times X \to R$ such that for all $x, y, z \in X$, we have: 

1. $d(x, y) \geq 0$ <br>
2. $d(x, y) = 0$ if and only if $x = y$
3. $d(x,y) = d(y, x)$
4. (Triangle inequality) $d(x, y) \leq d(x, z) + d(y, z)$


## Cauchy-Schwarz Inequality

__Theorem 1.__ For $z,  w \in C$, we have the following inequality:

1. (Cauchy-Schwarz Inequality) $|R(z \overline{w})| \leq |z||w|$
2. (Triangle inequality) $|z+w| \leq |z| + |w|$

_Proof._ On the right side, we have

\begin{aligned}
|z||w| & = |a+bi||c+di| \\
       & = \sqrt{a^2+b^2} \cdot \sqrt{c^2+d^2} \\
       & = \sqrt{a^2c^2 + a^2d^2 + b^2c^2 + b^2d^2} \\
\implies 
(|z||w|)^2 & = a^2c^2 + a^2d^2 + b^2c^2 + b^2d^2
\end{aligned}

On the left side, we have

\begin{aligned}
|R(z \overline{w})| & = |R((a+bi) (c-di)) | \\
                    & = |ac+bd| \\
\implies 
|R(z \overline{w})|^2 & = |ac+bd|^2 \\
                      & = a^2c^2 + 2acbd + b^2d^2           
\end{aligned}

It is easy to show that 

\begin{aligned}
(|z||w|)^2 - |R(z \overline{w})|^2 & = (ad)^2 + (bc)^2 - 2acbd \\
                                & = (ad-bc)^2 \\
                                & \geq 0  \\
\implies
|R(z \overline{w})| \leq |z||w|
\end{aligned}

For the triangle inequality, we have:

\begin{aligned}
 |z+w|^2 & = (z+w)(\overline{z} + \overline{w})  \\
        & = |z|^2 + |w|^2 + w\overline{z} + z\overline{w} \\
        & = |z|^2 + |w|^2 + w\overline{z} + \overline{(\overline{z})}\overline{w} \\
        & = |z|^2 + |w|^2 + w\overline{z} + \overline{(\overline{z}w)} \\
        & = |z|^2 + |w|^2  + 2 R(z\overline{w}) \\
        & \leq |z|^2 + |w|^2  + 2 |z\overline{w}| \\
        & = (|z| + |w|)^2 \\
\implies & |z+w|  \leq |z| + |w|
\end{aligned}

<span style="float:right">&#9726;</span>
<br>

For the general case of Cauchy–Schwarz inequality, please read [this paper](https://drive.google.com/file/d/1ofjPMh-006422bzSpeuC1OpNtiy9htSM/view?usp=sharing). 


## Sequences in Complex Numbers

__Definition 3.__ For a sequence of complex number $a_n$ and $L \in C$, to say
that $\lim_{n \to \infty} a_n = L$ means that for every $\epsilon > 0$, there
exists some $N(\epsilon) \in R$ such that if $n > N(\epsilon)$, then
$|a_n - L| < \epsilon$. For a given sequence $a_n$, if $\lim_{n \to \infty} a_n =L$
for some $L \in C$, then we say that $a_n$ _converges_, or is _convergent_;
otherwise, we say that $a_n$ _diverges_, or _divergent_. 

__Definition 4.__ To say that a nonempty subset $S$ of $C$ is _bounded_ means that
there exists some $M>0$ such that for $z \in S, |z| < M$. 

__Theorem 2.__ Let $a_n$ be a sequences in $C$.

1. If $a_n$ converges, then $a_n$ is bounded.
2. If $\lim_{n \to \infty} a_n = L \neq 0$, then there exits some real number
$K$ such that if $n > K$, then $|a_n| \geq \frac{|L|}{2}$.

_Proof._ 

(1.) If $a_n$ converges, then by definition, we could have some $N(\epsilon)$ 
such that if $n > N(\epsilon)$, then $|a_n - L| < \epsilon$. Now, choose
some integer $K > N(1)$, for $n > K$, we know that $|a_n -L| < 1$, which means
that 

\begin{equation}
|a_n| \leq |a_n - L| + |L| < |L| + 1 \ \ \ \ \text{(by the triangle inequality)}
\end{equation}

Therefore, since $\{ |a_1|, \cdots, |a_K| \}$ is a finite set, we see that for
all $n$, $|a_n| < M$, where $M = \max \{ |a_1|, \cdots, |a_K|, L+1 \}$. 

(2.) We know $\lim_{n \to \infty} a_n = L \neq 0$, we will prove $|a_n| > \frac{|L|}{2}$
when $n > K$ by contradiction. Let $\epsilon = \frac{|L|}{2} > 0$, since $a_n$
converges we have $|a_n - L| < \frac{|L|}{2}$ when $n > K$. Now we must have
$|a_n| \geq \frac{|L|}{2}$, otherwise we will have 

\begin{align}
|L| = |L - a_n + a_n|  \leq |L-a_n| + |a_n| < \frac{|L|}{2} +  \frac{|L|}{2} = |L|
\end{align}

$|L| < |L|$ is a contradiction. 
<span style="float:right">&#9726;</span>
<br>

??? tip
       Two dimensions of analysis:

       1. integer $N$
       2. measurement $\epsilon$

__Theorem 3.__ Let $a_n$ and $b_n$ be _sequences_ in $C$, and suppose that
$\lim_{n \to \infty} a_n = L, \lim_{n\to \infty} b_n = M$, and $c \in C$. Then
we have that 

1. $\lim_{n \to \infty} c a_ = c L$
2. $\lim_{n \to \infty} (a_n + b_n) = L + M$
3. $\lim_{n \to \infty} \overline{a_n} = \overline{L}$
4. $\lim_{n \to \infty} a_n b_n = LM$ 
5. if $L \neq  0$, then $\lim_{n \to \infty} \frac{1}{a_n} = \frac{1}{L}$
6. if $a_n$ is real-valued and $a_n \leq K$ for all $n$, then $\lim_{n \to \infty} a_n = L \leq K$. 

__Theorem 4.__ Let $z_n = x_n + y_n i$ be a _complex sequence_ with real and
imaginary parts $x_n$ and $y_n$, respectively, and let $L = a + bi \in C$ have
real and imaginary parts $a$ and $b$, respectively. Then $\lim_{n \to \infty} z_n = L$
if and only if $\lim_{n \to \infty} x_n = a$ and  $\lim_{n \to \infty} y_n = b$. 

## Completeness In Metric Spaces 