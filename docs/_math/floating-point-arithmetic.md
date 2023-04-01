---
title: Floating-Point Arithmetic
subtitle: Floating-Point arithmetic is a way of representing real numbers in a computer. It is a way of representing numbers in a computer that is not exact, but is fast and efficient. It is a fundamental concept in numerical computing.
layout: math_page_template
date: 2023-04-01
keywords: floating-point arithmetic python matrix
published: true
tags: algorithm data-science machine-learning numerical-linear-algebra python
---


Almost all computation in a computer is carried out using vectors and matrices. These are represented in a computer as arrays of numbers. The numbers in these arrays are usually real numbers. In order to represent real numbers in a computer, we need to use a representation that is not exact, but is fast and efficient. This is called floating-point arithmetic.

This concept is so fundamental to numerical computing that it is worth spending some time to understand it. There is a very famous article on floating-point arithmetic by David Goldberg, which is called _What Every Computer Scientist Should Know About Floating-Point Arithmetic_ {% cite goldberg1991every %}. It is a very good read, and I highly recommend it. However, it is a bit long, and it is not easy to understand.

In this article, we will discuss the basics of floating-point arithmetic, and how it is used in Python.

## Floating-Point 101

{% katexmm %}

Computers cannot represent arbitrary real numbers. They can only handle 
a bounded sequence of binary bits that can be interpreted as bounded integers. Therefore, numerical analysis could be define as:
_The study of algorithms for mathematical analysis that rely on calculating using bounded integers only_


In computer, I=integers are stored using a system called _Two's complement_. In this system,
$b$ bits are used to represent numbers between $-2^{b-1}$ and $2^{b-1}-1$.

__Floating-point numbers__. Floating-point numbers with base $2$ have the 
form 

$$\pm(1+\sum_{i=1}^{p-1}d_i2^{-i})2^e$$ 

where $e$ is called the exponent,
$p$ is the precision, and $1+\sum_{i=1}^{p-1}d_i2^{-i}$ is the significand (or mantissa).

We can also write the floating-point number as 

$$1.M \times 2^E$$

- $M$ = “mantissa” (or significand)
- $E$ = “exponent”


To get a feel for this, let's see an example: the number $3.140625$. First,
it is positive, so that sign is $+$. Second, since it is between $2$ and 
$2^2$, the exponent $e$ should be $1$. So in order for

$$3.140625 = (\text{significant)}2^1$$

We should have 

$$\text{significant} = 1.5703125$$

You can check that 

$$1.5703125 = 1+\sum_{i=1}^{p-1}d_i2^{-i} = 1+ \frac{1}{2} + \frac{1}{16} + \frac{1}{128} = 1+2^{-1} + 2^{-4} + 2^{-7}$$


```python
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
binary(3.140625)
# '01000000010010010000000000000000'
```

<div class='figure'>
    <img src="/math/images/floating-number.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The binary representation of the floating-point number $3.140625$.
    </div>
</div>

```python
# more examples
# function link
# https://gist.github.com/oceanumeric/23d741404e01fb57f720d31a3a8e8348
binary_decompostion(10)

#  Decimal: 1.25 x 2^3
#   Binary: 1.01 x 2^11
#     Sign: 0 (+)
# Mantissa: 01 (0.25)
# Exponent: 11 (3)

binary_decompostion(2.998e8)

#  Decimal: 1.11684203147888184 x 2^28
#   Binary: 1.0001110111101001010111 x 2^11100

#     Sign: 0 (+)
# Mantissa: 0001110111101001010111 (0.11684203147888184)
# Exponent: 11100 (28)
``` 

For `float32`, the exponent $e$ can be any number between $-127$ and $128$,
and the precision $p$ is $24$ (mantissa = 23). For `float64`, the exponent $e$ can be any number $-2047$ and $2048$ and the precision $p$ is $53$ (mantissa=52). Figure 2 gives the illustration of the binary representation of a floating-point number.


<div class='figure'>
    <img src="/math/images/floating-number-bits.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> 32-bit floating-point number representation and 64-bit floating-point number representation based on IEEE 754 standard.
    </div>
</div>


One notable feature of floating-point numbers is that __they are not distributed uniformly__
on the real axis. The diagram below shows the location of the finite numbers 
representable by a hypothetical 8-bit floating-point format that has a 
4-bit mantissa (with an implied fifth leading 1 bit) and a 3-bit exponent but 
otherwise follows IEEE conventions. _You will understand this distribution well after calculating different numbers with the code in this post_.

<div class='figure'>
    <img src="/math/images/floating-number-distributions.png"
         alt="floating number illustrated"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The distribution of floating-point numbers.
    </div>
</div>


To be super clear, the following gives the floating point
format written as powers of $2$ for all 52 mantissa bits:

$$2^{0}.2^{-1}2^{-2}2^{-3}\cdots 2^{-50}2^{-51}2^{-52} \times 2^e$$

Therefore, $2^{-52}$ is the smallest value we can add in the computer system. 
The smallest significant digit multiplied by our exponent gives us the spacing
between this number and the next number we can represent with our format.

```python
# mantissa times exponent
(2**-52) * (2**0)
# 2.220446049250313e-16
np.nextafter(1.0, 2)
# 1.0000000000000002
space = np.nextafter(1.0, 2)-1
space
# 2.220446049250313e-16
```

Notice the above result shows that the spacing between $1$ and $1.0000000000000002$ is $2.220446049250313e-16$, which is the same as $2^{-52}$.

```python
# space between 2 to 4
np.nextafter(2, 3)
# 2.0000000000000004
space = np.nextafter(2, 3)-2
space
# 4.440892098500626e-16
```

_Remark_: There are spaces between floating point numbers as it is shown in figure 3. Therefore, numbers will be rounded to the nearest number in the computer system. 


## Roundoff errors 


If you do a calculation that puts you somewhere between the space of two numbers, the computer will automatically round to the nearest one. I like to think of this as a mountain range in the computer. The valleys are representable numbers. If a calculation puts us on a mountain side, we’ll roll down the mountain to the closest valley. For example, spacing for the number 1 is $2^{-52}$, so if we don't add at least half that, we won't 
reach the next representable number (__therefore, this might give us roundoff errors__):

```python
# minimal space 
spacing = (2 ** -52) * (2 ** 0)
1 + 0.4 * spacing == 1  # add less than half the spacing
# True
1 + 0.6 * spacing == 1  # add more than half the spacing
# False
```

<div class='figure'>
    <img src="/math/images/floating-number-gaps.png"
         alt="floating number illustrated"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Illustration of the spacing between floating-point numbers.
    </div>
</div>

We’re working with pretty small numbers here so you might not be too shocked. 
But remember, the bigger our exponent, the bigger our rounding errors will be!

```python
# a very large number = 1e25
large_number = 1e25
binary_decompostion(large_number)

#  Decimal: 1.03397576569128469 x 2^83
#   Binary: 1.0000100010110010101000101100001010000000001010010001 x 2^1010011

#     Sign: 0 (+)
# Mantissa: 0000100010110010101000101100001010000000001010010001 (0.03397576569128469)
# Exponent: 1010011 (83)
# the exponent is 83, which is very large
spacing = (2 ** -52) * (2 ** 83)
print(spacing, f"{spacing:0.5e}")
# 2147483648.0 2.14748e+09
```

That's just over $2$ __billion__! $1$ billion is less than half that spacing, 
so if we add 1 billion to a _large number_,  we won’t cross the mountain peak, and we’ll slide back down into the same valley:

```python
one_billion = 1e9
1e25 + one_billion == 1e25
# True
```

The above test shows that the computer will round the number to the nearest number in the computer system even if we add a very large number to a very large number, which is completely wrong. 

2 billion is more than half that spacing, so if we add it to large_number, 
we’ll slide into the next valley (representable number):

```
two_billion = 2e9
1e25 + two_billion == 1e25
# False
```

Another thing you should know is that, just as with integer data types, there is a biggest and a smallest number you can represent with floating point. You could determine this by hand (by making the mantissa and the exponent as big/small as possible), but we’ll use NumPy:

```python
np.finfo("float64").max  # maximum representable number
# 1.7976931348623157e+308
np.finfo("float64").min  # minimum representable number
# -1.7976931348623157e+308
np.finfo("float64").tiny  # closest to 0
# 2.2250738585072014e-308
```

The spacing between floating-point numbers in the range $[1, 2)$ is $2.2 \times 10^{-16}$,
while it is $4.4 \times 10^{-16}$ in the range $[2, 4)$. 

We will denote 

$$u = \frac{1}{2} \times (\text{distance between 1 and the next floating-point number})$$

This is called the __unit roundoff__. In double precision, $u \approx 10^{-16}$.

Using floating-point arithmetic, $x+(y+z)$ is not necessarily equal to $(x+y)+z$ because the order of rounding is different.

```python
2.0 ** (-53)
# 1.1102230246251565e-16
(-1.0 + 1.0) + 2.0**(-53)  # get a very small value 2 ** (-53)
# 1.1102230246251565e-16
(-1.0) + (1.0 + 2.0**(-53))  # since 2 ** -53 < 2 ** -52, it was rounded off 
# 0.0
```

In numerical analysis, we often need to know the relative error of a calculation and we always want to have a stable algorithm, which means the catastrophic cancellation should be avoided.


In numerical analysis, catastrophic cancellation is the phenomenon that subtracting good approximations to two nearby numbers (which create roundoff error spaces because of gaps we illustrated) may yield a very bad approximation to the difference of the original numbers.

Catastrophic cancellation isn't affected by how large the inputs are—it applies just as much to large and small inputs. It depends only on how large the difference is, and on the error of the inputs.

Now, I hope you understand the concept of floating-point numbers and the spacing between them, and you can avoid catastrophic cancellation in your numerical analysis.

{% endkatexmm %}

