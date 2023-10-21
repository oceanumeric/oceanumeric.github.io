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

Here is the visualization of the Fourier Series of the square wave (when $N=100$, you can click the right bottom corner to see the animation):

<div align="center">
<iframe src="https://www.desmos.com/calculator/ztj3fj5qch?embed" width="500" height="300" style="border: 1px solid #ccc" frameborder=0></iframe>
</div>


From the above example,we can see that the fourier series is 'converging' to the square wave. The more terms we add, the more similar it is to the square wave. However, we 
also see _discontinuity_ at the jump points. This is called _Gibbs phenomenon_. Since
both sine and cosine are continuous, the fourier series of a function is also continuous. Therefore the fourier series of a discontinuous function will have discontinuity at the jump points.

Now, let's see another example - traingle wave - which is defined as:

$$
f(t) = \frac{1}{2} - |t| = \begin{cases} \frac{1}{2} + t & -\frac{1}{2} \leq t < 0 \\ \frac{1}{2} - t & 0 \leq t < \frac{1}{2} \end{cases}
$$

<div class='figure'>
    <img src="/math/images/triangle_wave.png"
         alt="triangle wave"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Illustration of triangle wave.
    </div>
</div>

The coefficient of $f(t)$ is at $n=0$ is the average of $f(t)$, which is $1/4$. For $n \neq 0$, we have:

$$
\begin{aligned}
c_n & = \int_{-1/2}^{1/2} f(t) e^{-2\pi i n t} dt \\
& = \int_{-1/2}^{1/2} (\frac{1}{2} - |t|) e^{-2\pi i n t} dt \\
& = \frac{1}{2} \int_{-1/2}^{1/2} e^{-2\pi i n t} dt - \int_{-1/2}^{1/2} |t| e^{-2\pi i n t} dt \\
& = - \int_{-1/2}^{1/2} |t| e^{-2\pi i n t} dt;  \quad \text{since } \int_{-1/2}^{1/2} e^{-2\pi i n t} dt = 0 \\
& =  - \bigg( \int_{-1/2}^{0} - t e^{-2\pi i n t} dt + \int_{0}^{1/2}  t e^{-2\pi i n t} dt \bigg) \\
& = \int_{-1/2}^{0} t e^{-2\pi i n t} dt - \int_{0}^{1/2}  t e^{-2\pi i n t} dt \\
\end{aligned}
$$

Now, let $A(n)$ be the first integral and we have:

$$
\begin{aligned}
A(n) & = \int_{-1/2}^{0} t e^{-2\pi i n t} dt 
\end{aligned}
$$

It is easy to show that

$$
\begin{aligned}
A(-n) & = \int_{-1/2}^{0} t e^{2\pi i n t} dt \\
& = \int_{1/2}^0 -s e^{-2\pi i n s} - ds \quad \text{let } s = -t \\
& =  \int_{1/2}^0 s e^{-2\pi i n s} ds \\
& = - \int_{0}^{1/2} s e^{-2\pi i n s} ds \\
& = - A(n) = - \int_{-1/2}^{0} t e^{-2\pi i n t} dt \\
\end{aligned}
$$

Therefore, the fourier coefficient can be written as:

$$
c_n = A(n) + A(-n)
$$

Now, let's integrate $A(n)$ by parts:

$$
\begin{aligned}
A(n) & = \int_{-1/2}^{0} t e^{-2\pi i n t} dt \\
& = \frac{1}{-2\pi i n} t e^{-2\pi i n t} \Big|_{-1/2}^0 - \int_{-1/2}^{0} \frac{1}{-2\pi i n} e^{-2\pi i n t} dt \\
& = \frac{1}{-2\pi i n} t e^{-2\pi i n t} \Big|_{-1/2}^0 - \frac{1}{(2\pi i n)^2} e^{-2\pi i n t} \Big|_{-1/2}^0 \\
& = \frac{1}{-2\pi i n} [0 + \frac{1}{2} e^{\pi i n}] - \frac{1}{(2\pi i n)^2} [1 - e^{\pi i n}] \\
& = - \frac{1}{4\pi i n}e^{\pi in} + \frac{1}{4\pi^2n^2} [1 - e^{\pi i n}] \\
& = \frac{\pi i n}{4 \pi^2 n^2} e^{\pi i n} + \frac{1}{4\pi^2n^2} [1 - e^{\pi i n}] \\
& = \frac{1}{4\pi^2n^2} [1 - e^{\pi i n} + \pi i n e^{\pi i n}] \\
& = \frac{1}{4\pi^2n^2} [ 1 + e^{\pi i n} (\pi i n - 1)] 
\end{aligned}
$$

Therefore, we could have 

$$
A(-n) = \frac{1}{4\pi^2n^2} [ 1 + e^{-\pi i n} (-\pi i n - 1)]
$$

The fourier coefficient is:

$$
\begin{aligned}
c_n & = A(n) + A(-n) \\
& = \frac{1}{4\pi^2n^2} [ 1 + e^{\pi i n} (\pi i n - 1)] + \frac{1}{4\pi^2n^2} [ 1 + e^{-\pi i n} (-\pi i n - 1)] \\
& = \frac{1}{4\pi^2 n^2} [ 2 + e^{\pi i n} (\pi i n - 1) + e^{-\pi i n} (-\pi i n - 1)] \\
& =  \frac{1}{4\pi^2 n^2} [ 2 + (\cos(\pi n) + i \sin(\pi n)) (\pi i n - 1) - (\cos(\pi n) - i \sin(\pi n)) (\pi i n + 1)] \\
& =  \frac{1}{4\pi^2 n^2} [ 2 + \cos(\pi n)(\pi in - 1) - \cos(\pi n)(\pi in +1)] \\
& = \frac{1}{2\pi^2 n^2} (1 - \cos(\pi n))  \\ 
& = \begin{cases} 0 & n \text{ is even} \\  \frac{1}{\pi^2 n^2} & n \text{ is odd}  \end{cases} 
\end{aligned}
$$

Now, let's write down the fourier series of $f(t)$:

$$
\begin{aligned}
f(t) & = \sum_{n=-\infty}^{\infty} c_n e^{2\pi i n t} \\
& = \sum_{n \text{ is odd}} \frac{1}{\pi^2 n^2} e^{2\pi i n t} \\
& = \sum_{-\infty}^{0} \frac{1}{\pi^2 n^2} e^{2\pi i n t} + \sum_{1}^{\infty} \frac{1}{\pi^2 n^2} e^{2\pi i n t} \\
& = c_{-n} e^{-2\pi i n t} + c_n e^{2\pi i n t} \\
& = c_n (e^{2\pi i n t} + e^{-2\pi i n t}) \\
& = \frac{2}{\pi^2 n^2} \cos (2 \pi n t) \\
& = \frac{1}{4} + \sum_{k=0}^\infty \frac{1}{\pi^2 (2k+1)^2} \cos[2\pi (2k+1) t] \\
\end{aligned}
$$

<div align="center">
<iframe src="https://www.desmos.com/calculator/xkhg6zaa9j?embed" width="500" height="300" style="border: 1px solid #ccc" frameborder=0></iframe>
</div>

For this example, there is no joumping points, so there is no Gibbs phenomenon. The fourier series is converging to the triangle wave. However, since we have infinite terms, the fourier series is not a triangle wave. It is a _smooth_ triangle wave. The fourier series is a smooth approximation of the triangle wave. The more terms we add, the more similar it is to the triangle wave.

This is due to the fact that the fourier series is a _linear combination_ of the basis $e^{2\pi i n t}$. The basis $e^{2\pi i n t}$ is a _smooth_ function, so the fourier series is also a smooth function. Or put it in another way, both sines and cosines are _differentiable_ to any order, so the fourier series is also differentiable to any order. 

In summary, __a discontinuoity in any order derivative of a periodic function will
force an infinite number of terms in the fourier series to approximate the function.__

Note also that for the triangle wave the coefficients decrease like $1/n^2$ while
for the square wave they decrease like $1/n$. Or, it takes around $N=100$ terms to approximate the square wave, but it only takes around $N=10$ terms to approximate the triangle wave. This has exactly do do wit the fact that the square wave is discontinuous while the triangle wave is continuous but its derivative is discontinuous.

<p class='theorembox'>
<b>Reflections</b>
<br>
I hope those two examples could give you the sense of how the fourier series works and how it converges to the original function in terms of the speed and the smoothness.
</p>

















{% endkatexmm %}