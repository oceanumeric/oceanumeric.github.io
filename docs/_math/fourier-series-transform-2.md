---
title: Fourier Series and Fourier Transform - II
subtitle: Certain feelings in my body lead me to believel that I have to stduy Fourier Series and Fourier Transform for a better understanding of probability theory, measure theory,entroy and information theory.
layout: math_page_template
date: 2023-10-15
keywords: fourier series, fourier transform, probability, measure theory, entropy, foundations of probability
published: true
tags: fourier-analysis probability measure-theory entropy foundations-of-probability
---

In our last [post](https://oceanumeric.github.io/math/2023/10/fourier-series-transform-1), we have introduced the big picture of Fourier Series. In this post, we will continue to explore the Fourier Series.


## Fourier Series

{% katexmm %}

For a periodic function $f(x)$ with period $1$ (you can normalize the period to $1$ by scaling the $x$-axis), we can write it as a Fourier Series:

$$
f(x) = \sum_{n=-\infty}^{\infty} c_n e^{2\pi i n x}
$$

where $c_n$ is the Fourier coefficient, which is the projection of $f(x)$ onto the basis $e^{2\pi i n x}$:

$$
c_n =\langle f(x), e^{2\pi i n x} \rangle =  \int_0^1 f(x) e^{-2\pi i n x} dx 
$$

I like using the inner product notation to represent the Fourier coefficient, because it is more intuitive to me. The inner product is the projection of $f(x)$ onto the basis $e^{2\pi i n x}$, which is the same as the Fourier coefficient.


Before we moving on, let's discuss the properties of the Fourier coefficient $c_n$:

1. $c_n$ is a complex number, which can be written as $c_n = a_n + i b_n$, where $a_n$ and $b_n$ are real numbers.
2. when $f(x)$ is a real function, $c_n = \overline{c_{-n}}$, where $\overline{c_{-n}}$ is the complex conjugate of $c_{-n}$. This is because the basis $e^{2\pi i n x}$ is a complex number, and the inner product of two complex numbers is a complex number. When $f(x)$ is a real function, the inner product of $f(x)$ and $e^{2\pi i n x}$ is a real number, which is the same as the inner product of $f(x)$ and $e^{-2\pi i n x}$.
3. $c_0$ is the average of $f(x)$, which is a real number:$$ c_0 = \int_0^1 f(x) dx$$

4. when $f(x)$ is even, $c_n$ is also even, which means $c_n = c_{-n}$. $$\begin{aligned}c_n & = \int_0^1 f(x) e^{-2\pi i n x} dx \\ & = - \int_0^{-1} f(-s) e^{2\pi i n s} ds  \quad \text{let } s = -x \\ & = \int_{-1}^0 f(-s) e^{-2\pi i n (-s)} ds \\ & = c_{-s} = c_{-n} \end{aligned} $$

5. when $f(x)$ is odd, $c_n$ is odd, which means $c_n = -c_{-n}$.

6. when $f(x)$ is real and even, $c_n$ is real and even, which means $c_n = c_{-n}$.

7. when $f(x)$ is real and odd, $c_n$ is imaginary and odd, which means $c_n = -c_{-n}$.

$$
c_n = -c_{-n} \quad \text{when } f(x) \text{ is odd} \\
c_n = \overline{c_{-n}} \quad \text{when } f(x) \text{ is real} \\
\longrightarrow - c_{-n} = \overline{c_{-n}} \quad \text{only possible when it is pure imaginary}
$$

Those properties are very useful when we calculate the Fourier coefficient as they could help us to verify the correctness of our calculation. Since most of signals are real, we can use those properties in practice.


## Two Examples

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

The Fourier coefficient of $f(x)$ is:

$$
\begin{aligned}
c_n & = \int_0^1 f(x) e^{-2\pi i n x} dx \\
& = \int_0^{\frac{1}{2}} e^{-2\pi i n x} dx - \int_{\frac{1}{2}}^1 e^{-2\pi i n x} dx \\
& = \frac{1}{-2\pi i n} e^{-2\pi i n x} \Big|_0^{\frac{1}{2}} - \frac{1}{-2\pi i n} e^{-2\pi i n x} \Big|_{\frac{1}{2}}^1 \\
& = \frac{1}{-2\pi i n}[e^{-\pi i n} - 1] - \frac{1}{-2\pi i n}[e^{-2\pi i n} - e^{-\pi i n}] \\
& = \frac{1}{-2\pi i n}[e^{-\pi i n} - 1 - e^{-2\pi i n} + e^{-\pi i n}] \\
& = \frac{1}{-2\pi i n}[2e^{-\pi i n} -1 - (\cos(-2\pi n) + i \sin(2\pi n))] \\
& = \frac{1}{-2\pi i n}[2e^{-\pi i n} -1 - 1] \\
& = \frac{1}{-2\pi i n}[2e^{-\pi i n} -2] \\
& = \frac{1}{-\pi i n}[e^{-\pi i n} -1] \\
& = \frac{1}{\pi i n}[1- e^{-\pi i n}] \\
\end{aligned}
$$

Therefore, the fourier series of $f(x)$ is:

$$
\begin{aligned}
f(x) & = \sum_{n=-\infty, n\neq 0}^{\infty} c_n e^{2\pi i n x} \\
& = \sum_{n=-\infty, n \neq 0}^{\infty} \frac{1}{\pi i n}(1- e^{-\pi i n}) e^{2\pi i n x} 
\end{aligned}
$$

Notice, $f(x)$ is an odd function, so $c_n$ is imaginary and odd, which means $c_n = -c_{-n}$. Notice that

$$
\begin{aligned}
1 - e^{-\pi i n} & = 1 - \cos(\pi n) - i \sin(\pi n) \\
&  = 1 - (-1)^n - i \sin(\pi n) \\
& = \begin{cases} 0 & n \text{ is even} \\ 2 & n \text{ is odd} \end{cases}
\end{aligned}
$$

So the series can be simplified as:

$$
\begin{aligned}
f(x) & = \sum_{n=-\infty, n\neq 0}^{\infty} c_n e^{2\pi i n x} \\
& = \sum_{n \text{ is odd}} \frac{2}{\pi i n}e^{2\pi i n x} \\
\end{aligned}
$$

<p class='theorembox'>
<b>Reflections</b>
<br>
We have shown that when the function is real and odd, the fourier coefficients 
are pure imaginary and odd.
</p>

Now, we combine the positive and negative terms together:

$$
e^{2\pi i n x} - e^{-2\pi i n x} = 2i \sin(2\pi n x)
$$

let $n = 2k+1$, we have:

$$
\begin{aligned}
f(x) & = \sum_{n \text{ is odd}} \frac{2}{\pi i n}e^{2\pi i n x} \\
& = \sum_{-\infty}^{\infty} \frac{2}{\pi i (2k+1)}e^{2\pi i (2k+1) x} \\
& = \sum_{-\infty}^{\infty} \frac{2}{\pi i (2k+1)}(e^{2\pi i (2k+1) x} - e^{-2\pi i (2k+1) x}) \\
& = \sum_{k=0}^{\infty} \frac{4}{\pi (2k+1)} \sin[2\pi (2k+1) x] \\
& = \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{1}{2k+1} \sin[2\pi (2k+1) x] 
\end{aligned}
$$

Here is the visualization of the Fourier Series of the square wave:

<iframe src="https://www.desmos.com/calculator/sgxedvuhnn?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>



















{% endkatexmm %}