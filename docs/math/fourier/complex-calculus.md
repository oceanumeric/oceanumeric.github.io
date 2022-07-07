---
title: Complex-valued Calculus 
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


Having established the fundamentals of real and complex numbers in the last
post, in this post, we will review key concepts of calculus in complex domain. 

## Continuity and Limits

__Definition 1.__ Let $X$ be a nonempty subset of $C$, let $f: X \to C$ be a
function, and let $a$ be a point in $X$. To say that $f$ is _continuous_ at $a$
means that one of the following conditions hold:

1. (_sequential continuity_) For every sequence $z_n$ in $X$ such that
$\lim_{n \to \infty} z_n = a$, we have that $\lim_{n \to \infty} f(z_n) = f(a)$. 

2. ($\epsilon$-$\delta$ continuity) For every $\epsilon > 0$, there exists some
$\delta(\epsilon) > 0$ such that if $|z - a| < \delta(\epsilon)$, then
$|f(z)-f(a)| < \epsilon$. 

To say $f$ is _continuous_ on $X$ means that $f$ is continuous at $a$ for all
$a \in X$. 

__Theorem 2.__ Let $X$ and $Y$ be _metric space_. Let $f: X \to Y$ be a function,
and let $a$ be a point in $X$. Then $f$ is sequentially continuous at $a$ if and
only if $f$ is $\epsilon$-$\delta$ continuous at $a$. 

We will find both the sequential and $\epsilon$-$\delta$ continuity continuity
to be useful for different purposes. For example, $\epsilon$-$\delta$ continuity
will later be useful when considering the properties of a continuous function
over an entire interval. Conversely, sequential continuity makes the proof of
the algebraic properties of continuity straightforward. 

__Theorem 3.__ Let $X$ be a subset of $C$. Let $f, g : X \to C$ be functions,
and for some $a \in X$, suppose that $f$ and $g$ are continuous at $a$. Then:

1. for $c \in C$, $cf(z)$ is continuous at $a$.
2. $f(z) + g(z)$ is continuous at $a$.
3. $\overline{f(z)}$ is continuous at $a$
4. $f(z)g(z)$ is continuous at $a$
5. if $g(z) \neq 0$ for all $z in X$, then $f(z)/g(z)$ is continuous at $a$. 

__Theorem 4.__ Let $X, Y$ and $Z$ be _metric spaces_, let $f: X \to Y$ and 
$g : Y \to Z$ be functions, let $a$ be a point in $X$, and suppose that $f$
is continuous at $a$ and $g$ is continuous at $f(a)$. Then $g \circ f$ is
continuous. 

__Example 5.__ For $X = R$ or $C$, any polynomial function $p: X \to X$ is
continuous. 

__Definition 6.__ Let $X$ be a nonempty subset of $C$ and let $f: X \to C$ be a
function. To say $f$ is _uniformly continuous_ on $X$ means that for every
$\epsilon > 0$, there exists some $\delta(\epsilon) >0$ such that if $z, w \in X$
and $|z - w| < \delta(\epsilon)$, then $|f(z) - f(w)| < \epsilon$. 


!!! note
    Uniform continuity on $X$ is therefore generally a stricter condition than
    continuity on $X$, as the "degree of continuity" $\delta(\epsilon)$ is
    no longer allowed to vary with $a$ (point-by-point). Obviously if a function is uniformly 
    continuous it is continuous. But not necessarily vice versa—sometimes the
    function is such that the $\delta$ must depend on the actual point you’re going to.

__Example 8.__ The function $g(x) = \sqrt{x}$ is uniformly continuous on $[0, \infty]$.
For given $\epsilon$, let $\delta = \epsilon^2$, then if $|x - c| < \delta = \epsilon^2$,
we have $-\delta < x -c < \delta$. It gives

\begin{aligned}
& x <  c + \delta = c + \epsilon^2 < (\sqrt{c} + \epsilon)^2 \implies \sqrt{x} < \sqrt{c} + \epsilon  \\
& x > c - \delta \implies \sqrt{x} > \sqrt{c} + \epsilon 
\end{aligned}

This means we have $|\sqrt{x} - \sqrt{c}| < \epsilon$. So $|g(x) - g(c)| < \epsilon$,
which gives the uniformly continuous. 

__Example 9.__ The function $g(x) = \sin(\frac{1}{x})$ is not uniformly continuous
on $(0, 1)$. This is because the function oscillates so fast that for any 
$\delta$ you might pick, there is an x close to zero so that the interval 
$(x, x+\delta)$ contains an entire “period” of the function, and gets outside of 
any $\epsilon$ range whenever $\epsilon < 1$ (see the following figure). 


![sinfun](./images/sinfun.png)


__Theorem 7.__ If $X$ is a closed and bounded subset of $C$ and $f: X \to C$ is
continuous, then $f$ is _uniformly continuous_ on $X$. 

_Proof._ Assume otherwise. Then for _some_ $\epsilon > 0$ and _every_ $\delta > 0$
there exists $x, y$ such that $|x-y| < \delta$ but $|f(x) - f(y)| > \epsilon$. 
For $\delta=1 / n$ choose such an $x$ and $y$ and call them $x_{n}$ and $y_{n}$ 
respectively. Since the interval of interest is closed and bounded, there is a 
subsequence $\{x_{n_{k}}\}$ of $\{x_{n}\}$ that converges. Call its limit $c$. 
Certainly $c \in[a, b]$. And we have,

$$|y_{n_{k}}-c|=|y_{n_{k}}-x_{n_{k}}+x_{n_{k}}-c|<|y_{n_{k}}-x_{n_{k}}|+|x_{n_{k}}-c|$$

and both terms go to zero, so the $y_{n_{k}}$ also converge to $c$. Now 

\begin{aligned}
|f(x_{n_{k}})-f(c)| & =|f(x_{n_{k}})-f(y_{n_{k}})+f(y_{n_{k}})-f(c)|  \\
                    & \geq|f(x_{n_{k}})-f(y_{n_{k}})|-\mid f(y_{n_{k}})-f(c)| \\
                    & \geq \epsilon-| f(y_{n_{k}})-f(c) 
\end{aligned}

If $f$ were continuous at $c$, both terms in absolute values would go to zero, raising a contradiction. So $f$ is not continuous.
<span style="float:right">&#9726;</span>
<br>
