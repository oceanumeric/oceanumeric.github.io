---
title: Fourier Series and Fourier Transform - III
subtitle: Certain feelings in my body lead me to believel that I have to stduy Fourier Series and Fourier Transform for a better understanding of probability theory, measure theory,entroy and information theory.
layout: math_page_template
date: 2023-12-31
keywords: fourier series, fourier transform, probability, measure theory, entropy, foundations of probability
published: true
tags: fourier-analysis probability measure-theory entropy foundations-of-probability
---

This is the third part of the series on Fourier Series and Fourier Transform. In the
first two parts, we have discussed the Fourier Series. From this part onwards, we will
forcus on the Fourier Transform. In a nutshell, Fourier transform could be thought of
as the extension of Fourier Series to the continuous domain, which means that we could
extend the Fourier Series to the functions which are not periodic.


## Review of Fourier Series

{% katexmm %}

For a periodic function $f(x)$ with period $1$ (you can normalize the period to $1$ by scaling the $x$-axis), we can write it as a Fourier Series:

$$
f(x) = \sum_{n=-\infty}^{\infty} c_n e^{2\pi i n x}
$$

where $c_n$ is the Fourier coefficient, which is the projection of $f(x)$ onto the basis $e^{2\pi i n x}$:

$$
c_n =\langle f(x), e^{2\pi i n x} \rangle =  \int_0^1 f(x) e^{-2\pi i n x} dx 
$$

where $c_n$ is the Fourier coefficient, which is the projection of $f(x)$ onto the basis $e^{2\pi i n x}$. 


Conside a _square wave_ of period $1$, which is defined as:

$$
f(x) = \begin{cases} 1 & 0 \leq x < \frac{1}{2} \\ -1 & \frac{1}{2} \leq x < 1 \end{cases}
$$

<div class='figure'>
    <img src="/math/images/square_wave.png"
         alt="square wave"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Illustration of square wave.
    </div>
</div>

We have calculated the Fourier coefficients of the square wave in the previous part:

$$
\begin{aligned}
f(x) & = \sum_{n \text{ is odd}} \frac{2}{\pi i n}e^{2\pi i n x} \\
& = \sum_{-\infty}^{\infty} \frac{2}{\pi i (2k+1)}e^{2\pi i (2k+1) x} \\
& = \sum_{-\infty}^{\infty} \frac{2}{\pi i (2k+1)}(e^{2\pi i (2k+1) x} - e^{-2\pi i (2k+1) x}) \\
& = \sum_{k=0}^{\infty} \frac{4}{\pi (2k+1)} \sin[2\pi (2k+1) x] \\
& = \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{1}{2k+1} \sin[2\pi (2k+1) x] 
\end{aligned}
$$

The above formula shows the fourier coefficients of the square wave is zero for even $n$ and non-zero for odd $n$. And each component of the Fourier Series is a sine wave with frequency $2k+1$. The following figure shows how the square wave is constructed by the Fourier Series. 

<div class='figure'>
    <img src="/math/images/fourier_series-011.png"
         alt="fourier series"
         style="width: 79%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Illustration of Fourier Series (Figure was taken from TikZ.net by Izaak Neutelings).
    </div>
</div>


This gives us a hint that we can use Fourier Series to decompose a periodic function into a series of sine waves with different frequencies. 


{% endkatexmm %}