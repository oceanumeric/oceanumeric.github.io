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


__Definition 2.__ A metric space \( (\mathcal{M}, d) \) is a set \( \mathcal{M} \) 
together with a function \( d: \mathcal{M} \times \mathcal{M} \to \mathbb{R} \) 
called a metric satisfying four conditions:

1. $d (x, y ) \geq 0$ for all $x, y \in \mathcal{M}$. (non-negativeness)
2. $d(x, y) = 0$ if and only if $x = y$. (identification)
3. $d(x, y) = d(y, x)$ for all $x, y \in \mathcal{M}$. (symmetry)
4. $d(x, y) \leq d (x, z) + d(z, y)$  for all $x, y, z \in \mathcal{M}$. (triangle inequality)



## Cauchy-Schwarz Inequality

__Theorem 3.__ For $z,  w \in C$, we have the following inequality:

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

__Definition 4.__ For a sequence of complex number $a_n$ and $L \in C$, to say
that 

$$\lim_{n \to \infty} a_n = L$$ 

means that for every $\epsilon > 0$, there
exists some $N(\epsilon) \in R$ such that

$$n > N(\epsilon) \implies |a_n - L| < \epsilon$$

For a given sequence $a_n$, if $\lim_{n \to \infty} a_n =L$
for some $L \in C$, then we say that $a_n$ _converges_, or is _convergent_;
otherwise, we say that $a_n$ _diverges_, or _divergent_. 

__Definition 5.__ To say that a nonempty subset $S$ of $C$ is _bounded_ means that
there exists some $M>0$ such that for $z \in S, |z| < M$. 

__Theorem 6.__ Let $a_n$ be a sequences in $C$.

1. If $a_n$ converges, then $a_n$ is bounded.
2. If $\lim_{n \to \infty} a_n = L \neq 0$, then there exits some real number
$K$ such that if $n > K$, then $|a_n| \geq \frac{|L|}{2}$.

_Proof._ (1.) If $a_n$ converges, then by definition, we could have some $N(\epsilon)$ 
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

__Theorem 7.__ Let $a_n$ and $b_n$ be _sequences_ in $C$, and suppose that
$\lim_{n \to \infty} a_n = L, \lim_{n\to \infty} b_n = M$, and $c \in C$. Then
we have that 

1. $\lim_{n \to \infty} c a_ = c L$
2. $\lim_{n \to \infty} (a_n + b_n) = L + M$
3. $\lim_{n \to \infty} \overline{a_n} = \overline{L}$
4. $\lim_{n \to \infty} a_n b_n = LM$ 
5. if $L \neq  0$, then $\lim_{n \to \infty} \frac{1}{a_n} = \frac{1}{L}$
6. if $a_n$ is real-valued and $a_n \leq K$ for all $n$, then $\lim_{n \to \infty} a_n = L \leq K$. 

__Theorem 8.__ Let $z_n = x_n + y_n i$ be a _complex sequence_ with real and
imaginary parts $x_n$ and $y_n$, respectively, and let $L = a + bi \in C$ have
real and imaginary parts $a$ and $b$, respectively. Then $\lim_{n \to \infty} z_n = L$
if and only if $\lim_{n \to \infty} x_n = a$ and  $\lim_{n \to \infty} y_n = b$. 

## Completeness In Metric Spaces 

__Definition 9.__ Let $a_n$ be a complex-valued sequence $a_n$. To say that
$a_n$ is _Cauchy_ means that for every $\epsilon > 0$, there exists some 
$N(\epsilon) \in R$ such that if $n, k > N(\epsilon)$, then $|a_n- a_k| < \epsilon$.
More generally, if $a_n$ is a sequence in a metric space $X$, so that $a_n$ is 
Cauchy means that for every $\epsilon > 0$, there exists some $N(\epsilon) \in R$
such that if $n, k > N(\epsilon)$, then $d(a_n, a_k) < \epsilon$. 

__Theorem 10.__ Let $a_n$ be a _convergent sequence_ in a metric space $X$. Then
$a_n$ must be _Cauchy_. 

_Proof._ If $\lim_{n \to \infty} a_n = L$, then by definition, we have

$$d(a_n, L) < \epsilon$$

when $n > N(\epsilon)$. Therefore, we could have

$$d(a_n, a_k) \leq d(a_n, L) + d(a_k, L) < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon$$

when $n, k > N'(\frac{\epsilon}{2})$. 
<span style="float:right">&#9726;</span>
<br>

__Example 11.__ Let $X = Q$, with the usual metric $d(x, y) = |x-y|$, and let
$a_n$ be a rational-valued sequence whose limit is irrational. Then by Theorem 5,
$a_n$ is Cauchy, as it converges in $R$, but $a_n$ does not have a limit in
$X = Q$. 

Completeness is the key property of the real numbers and complex numbers that 
the rational numbers lack. 

__Theorem 12.__ Between any two distinct real numbers there is an irrational number.

_Proof._ Consider the set of numbers of the form \(\frac{p}{q}\) with \(q\) fixed, 
and \(p\) any integer. Assume that there are no such numbers between real numbers
\(a\) and \(b\). Let \(\frac{p}{q}\) be the number immediately before \(a\). 
Then \( \frac{p+1}{q} \) is the number immediately after \(b\). We necessarily have 

$$\frac{p+1}{q} - \frac{p}{q} \geq b - a \ \ \Longleftrightarrow \frac{1}{q} \geq b - a$$

If we choose \(q\) sufficiently large, then the above inequality is wrong. 
Then there is at least one rational number between \(a\) and \(b\).
<span style="float:right">&#9726;</span>
<br>

__Definition 13.__ A metric space \( (X, d) \) is _Cauchy complete_ if any of the 
following equivalent conditions are satisfied:

1. Every Cauchy sequence of points in \(X \) has a limit that is also in \(X \)
2. Every Cauchy sequence in \(X \) converges in \(X \) (that is , to some point of \(X\)).

__Lemma 14.__ If $a_n$ is a Cauchy sequence in $C$, then $a_n$ is bounded

__Theorem 15.__ (Bolzano-Weierstrass). Every bounded sequence in $R$ has a convergent
subsequence. 

__Theorem 16.__ The real numbers are a complete metric space. 

__Corollary 17.__ The complex numbers are a complete metric space. 

__Theorem 18.__ (Bolzano-Weierstrass in $C$) every bounded sequence in $C$ has a 
convergent subsequence. 
								