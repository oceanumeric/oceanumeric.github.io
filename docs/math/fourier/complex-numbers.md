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

The _norm_ of $a+bi$ is defined to be 

$$|a+bi| = \sqrt{a^2+b^2}$$


For $z, w \in C$, we have:

* $\overline{z+w} = \overline{z} + \overline{w}$ 
* $\overline{z\cdot w} = \overline{z} \cdot \overline{w}$ 
* $|zw| = |z| |w|$ 


__Definition 2.__ Let $X$ be a nonempty set. A _metric_ on $X$ is a function
$d: X \times X \to R$ such that for all $x, y, z \in X$, we have: 

1. $d(x, y) \geq 0$ <br>
2. $d(x, y) = 0$ if and only if $x = y$
3. $d(x,y) = d(y, x)$
4. (Triangle inequality) $d(x, y) \leq d(x, z) + d(y, z)$


## Cauchy-Schwarz Inequality

__Theorem 1.__ For $z,  w \in C$, we have the following inequality:

$$|R(z \overline{w})| \leq |z||w|$$

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




