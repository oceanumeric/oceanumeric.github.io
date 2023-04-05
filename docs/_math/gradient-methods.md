---
title: Gradient Methods
subtitle: Gradient descent is one of the most popular optimization algorithms in machine learning. It can be used for both convex and non-convex optimization problems. In this post, we will learn about the key ideas behind gradient descent and how it can be used to solve optimization problems.
layout: math_page_template
date: 2023-04-04
keywords: calculus, gradient methods, optimization, machine learning, data science
published: true
tags: optimization machine-learning data-science
---

This post is based on a course from University of California, Berkeley - [Convex Optimization and Approximation](https://ee227c.github.io/){:target="_blank"}. I summarize the key ideas from the course and provide some examples to help you understand the concepts better.


Gradient descent at core is a form of _local search_. It starts with an initial guess and iteratively improves the guess by moving in the direction of the negative gradient. The size of the step is controlled by a parameter called the _learning rate_. The algorithm stops when the gradient is zero or when the change in the guess is below a certain threshold. The algorithm is visualized below. 

<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/JpJn4SQ8e0E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

For the broad class of convex function, gradient descent is guaranteed to converge to the _global minimum_. For non-convex functions, it is possible that the algorithm converges to a _local minimum_. In practice, it is often used to find a good local minimum.

- [Convexity](#convexity)
- [Gradient Descent](#gradient-descent)
- [Some applications of gradient descent](#some-applications-of-gradient-descent)


## Convexity

{% katexmm %}


A set is convex if for any two points in the set, the line segment connecting them also lies in the set. A function is convex if for any two points on the function, the line segment connecting them also lies above the function.

For a convex set $S \in \mathbb{R}^n$, for all $x, y \in S$, and all scalar $\alpha \in [0, 1]$, we have:

$$
\alpha x + (1 - \alpha)y \in S \tag{1}
$$

A function $f: \Omega \to \mathbb{R}$ is convex if for all $x, y \in \Omega$ and all scalar $\alpha \in [0, 1]$, we have:

$$
f(\alpha x + (1 - \alpha)y) \leq \alpha f(x) + (1 - \alpha)f(y) \tag{2}
$$

The _epigraph_ of a convex function $f$ is the set of all points $(x, y)$ such that $y \geq f(x)$, which means 

$$
\mathrm{epi}(f) = \{(x, t) | f(x) \leq t\} \tag{3}
$$

The epigraph of a convex function is always a convex set.

<div class='figure'>
    <img src="/math/images/Epigraph.svg"
         alt="Inequality bounds compare"
         style="width: 40%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of epigraph of a convex function (the figure was taken from wikipedia).
    </div>
</div>

We define the _gradient_ of a function $f: \Omega \to \mathbb{R}$ as the vector of partial derivatives:

$$
\nabla f(x) = \left(\frac{\partial f}{\partial x_i}\right)_{i=1}^n \tag{4}
$$

According to the definition of derivative, we have

$$
\nabla f(x)^T y = \frac{\partial f(x + \delta y)}{\partial \delta } \tag{5}
$$

According to Taylor's theorem, we have

$$
f(x + \delta ) = f(x) + \delta \nabla f(x)^T  + \frac{1}{2} \delta^2  \nabla^2 f(x)  + \mathcal{O}(\delta^3) \tag{6}
$$

We define the _Hessian_ of a function $f: \Omega \to \mathbb{R}$ as the matrix of second-order partial derivatives:

$$
\nabla^2 f(x) = \left(\frac{\partial^2 f}{\partial x_i \partial x_j}\right)_{i, j = 1}^n \tag{7}
$$

Schwarz’s theorem implies that the Hessian at a point $x$ is symmetric and positive definite if and only if $f$ is continuous and have second derivative at $x$.

Much of this post will be about different ways of minimizing a convex function $f: \Omega \to \mathbb{R}$, where $\Omega$ is a convex set:

$$
\min_{x \in \Omega} f(x) \tag{8}
$$

_Remark_: convex optimization is not necessarily easy to solve (even though the mathematical concepts are very intuitive). For starters, convex sets do not necessarily have a nice representation or enjoy compact descriptions. 

For example, the set of all matrices with non-negative entries is convex, but it is not easy to represent it in a computer. Secondly, the problem of finding the minimum of a convex function is not necessarily easy to solve. For example, the problem of finding the minimum of a quadratic function is easy to solve, but the problem of finding the minimum of a quadratic function with a linear constraint is not easy to solve.

When solving computational problems involving convex sets, we need to worry about how to represent the convex set we’re dealing with. Rather than asking for an explicit description of the set, we can instead require a computational abstraction that highlights essential operations that we can carry out. The Separation
Theorem motivates an important computational abstraction called _separation oracle_.

A _separation oracle_ for a convex set $K$ is a function that takes as input a point $x \notin K$ returns a hyperplane separating $x$ from $K$.  The separation oracle is a computational abstraction that allows us to solve convex optimization problems without having to explicitly represent the convex set $K$. 

<div class='figure'>
    <img src="/math/images/separation_oracle.png"
         alt="Inequality bounds compare"
         style="width: 50%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The illustration of
        separation oracle (you can understand the separation oracle as a process of finding boundary of a convex set).
    </div>
</div>


__Algorithm efficient__: Our goal of minimizing a convex function $f$ is equivalent to finding a point $x$ such that

$$
f(x) \leq \min_{x \in \Omega} f(x) + \epsilon, \quad \forall \ \epsilon > 0
$$

The cost of algorithm depends on our tolerance $\epsilon$. Highly practical algorithms often have polynomial time complexity in $\epsilon$, 
such as  $\mathcal{O}(1/\epsilon)$. or even $\mathcal{O}(1/\epsilon^2)$.


In this series of posts, we will make an attempt to highlight both the theoretical performance and practical appeal of an algorithm. Moreover, we will discuss other performance
criteria such as robustness to noise. How well an algorithm performs is rarely decided
by a single criterion, and usually depends on the application at hand.


## Gradient descent

For a differentiable function $f: \Omega \to \mathbb{R}$, the basic gradient descent method taking an initial point $x_0$ is defined as

$$
x_{t+1} = x_t - \eta_t \nabla f(x_t), \quad t = 0, 1, 2, \cdots \tag{9}
$$

where the sclae $\eta_t$ is the so-called _learning rate_ or (_step size_). The learning rate is a hyperparameter that controls the size of the step we take at each iteration. It may vary from iteration to iteration, or it may be fixed.

There are numerous ways of choosing step sizes that have a significant effect on
the performance of gradient descent. What we will see in this lecture are several choices
of step sizes that ensure the convergence of gradient descent by virtue of a theorem.
These step sizes are not necessarily ideal for practical applications.


In equation (9), $x_{t+1}$ might not be in $\Omega$ (watch the video at the beginning of the post, we cannot guarantee all iterations will stay within the domain). In this case, we can use the separation oracle to find a hyperplane that separates $x_{t+1}$ from $\Omega$ and project $x_{t+1}$ onto the hyperplane. This is called _projection_.

For a point $\tilde{x}$ that is not in $\Omega$, the projection of $\tilde{x}$ onto $\Omega$ is defined as

$$
\mathrm{proj}_{\Omega}(\tilde{x}) = \arg \min_{x \in \Omega} \|x - \tilde{x}\|_2^2 \tag{10}
$$

<div class='figure'>
    <img src="/math/images/gradient-projection.png"
         alt="Inequality bounds compare"
         style="width: 50%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The illustration of projection (figure was taken from Patrick Rebeschini).
    </div>
</div>

Now we are guaranteed that $x_{t+1}$ is in $\Omega$ for all iterations.
Note that computing the projection may be
computationally the hardest part of the problem. However, there are convex sets for which we know explicitly how to compute the projection. 


Here is our new gradient descent algorithm:

- Starting from $x_0$, repeat the following steps until convergence:
    - $x_{t+1} = x_t - \eta_t \nabla f(x_t)$
    - $x_{t+1} = \mathrm{proj}_{\Omega}(x_{t+1})$


When we implement the above algorithm, there are many practical issues we have to deal with. One of the most important issues is to make sure that the algorithm converges. If the gradients of objective function are too large, the algorithm may diverge. If the gradients are too small, the algorithm may take too many iterations to converge (see the following video to get some feeling about the convergence of gradient descent).

<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/VLlRqUK8ffA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

To help us to quantify the convergence of gradient descent, we need to introduce the notion of _gradient Lipschitz constant_.

A function $f: \Omega \to \mathbb{R}$ is _Lipschitz continuous_ with constant $L$ if for all $x, y \in \Omega$, we have

$$
\|f(x) - f(y)\| \leq L \|x - y\| \tag{11}
$$

If the function $f$ is L-Lipschitz continuous, and convext, then

$$
||\nabla f(x) || \leq L \tag{12}
$$

which gives a loose bound on the norm of the gradient of $f$. We have a more formal version of this bound in the following theorem.

__Theorem 1__. Let $f: \Omega \to \mathbb{R}$ be a convex function that is L-Lipschitz continuous. Let $R$ be the upper bound on the distance $\|x - x^*\|$ between the initial point $x_0$ and the optimal point $x^*$. Let $x_1, \cdots, x_t$ be the sequence generated by gradient descent with constant learning rate $\eta = \frac{R}{L\sqrt{t}}$. Then,  


$$
f\left ( \frac{1}{t} \sum_{s=1}^t x_s \right) - f(x^*) \leq \frac{RL}{\sqrt{t}} \tag{13}
$$

This means that the difference between the functional value of the average point
during the optimization process from the optimal value is bounded above by a constant
proportional to $1/\sqrt{t}$. This is a very useful result, because it tells us that the convergence of gradient descent is very fast. In fact, the convergence rate is $\mathcal{O}(1/\sqrt{t})$.

The proof of Theorem 1 is not difficult, but it is a bit tedious. We will skip the proof here. If you are interested, you can find the proof in the book _Convex Optimization_ by Boyd and Vandenberghe.


After learning the bound on the convergence rate, we now discuss the _smoothness_ of the objective function.  

A continuously differentiable function $f: \mathbb{R}^n \to \mathbb{R}^n$ is 
$\beta$ _smooth_ if the gradient map $\nabla f$ is $\beta$-Lipschitz continuous, meaning

$$
\| \nabla f(x) - \nabla f(y) \| \leq \beta \|x - y\| \tag{14}
$$

_Why do we care about the smoothness of the objective function?_  In the gradient descent algorithm, we need to decide the step size $\eta$. If the gradient changes too quickly, we need to adjust the step size quickly as well.The smoothness of the objective function tells us how quickly the gradient changes. In summary, for the objective function, we care two things:

- the Lipschitz constant of the objective function
- the smoothness of the objective function

The first one gives the range of gradient values, and the second one tells us how quickly the gradient changes.


__Lemma 1.__ Let $f$ be a f $\beta$-smooth function. Then, for any $x, y \in \mathbb{R}^n$, we have

$$
\left | f(y) - f(x) - \nabla f(x)^T (y - x) \right | \leq \frac{\beta}{2} \|y - x\|^2 \tag{15}
$$

The significance of this lemma is that we can choose $ y = x - \frac{1}{\beta} \nabla f(x)$ to minimize the right hand side of the above inequality. This gives us:

$$
f(y) - f(x) \leq - \frac{1}{2\beta} || \nabla f(x) ||^2 \tag{16}
$$

This means that the gradient update decreases the function value by an amount proportional to the squared norm of the gradient. This is a very useful result, because it tells us that the gradient descent algorithm is guaranteed to decrease the function value at each iteration.


__Lemma 2.__ Let $f$ be a $\beta$-smooth function. Then, for any $x, y \in \mathbb{R}^n$, we have

$$
f(x) - f(y) \leq \nabla f(x)^T (x - y) + \frac{\beta}{2} \| x - y  \|^2 \tag{17}
$$

We will show that gradient descent with the update rule

$$
x_{t+1} = x_t - \eta_t \nabla f(x_t) 
$$

attains a faster convergence rate under the smoothness condition. 

__Theorem 2.__ Let $f$ be a $\beta$-smooth function on $\mathbb{R}^n$ then gradient descent with the step size $\eta_t = \frac{1}{\beta t}$ 
satisfies 

$$
f(x_t) - f(x^*) \leq \frac{2 \beta || x_1 - x^* ||^2}{t-1} \tag{18}
$$

which means that the convergence rate of gradient descent is $\mathcal{O}(1/t)$.


Now, we will introduce the notion of strong convexity and combines it with smoothness
to develop the concept of condition number. While smoothness gave as an upper bound
on the second-order term in Taylor’s approximation, strong convexity will give us a lower bound. Taking together, those two assumptions are very powerful as they lead to a much faster convergence rate of the form $\exp(-\alpha t)$, where $\alpha$ is a constant.


A function $f: \Omega \to \mathbb{R}$ is _strongly convex_ with constant $\alpha > 0 $ if for all $x, y \in \Omega$, we have

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x) + \frac{\alpha}{2} \|y - x\|^2 \tag{19}
$$

or in different expression

$$
f(y) - f(x) \geq \nabla f(x)^T (y - x) + \frac{\alpha}{2} \|x - y\|^2 \tag{20}
$$


<div class='figure'>
    <img src="/math/images/smooth-and-strong.png"
         alt="Inequality bounds compare"
         style="width: 50%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The illustration of $\beta$-smooth and $\alpha$-strong convexity (figure was taken from Francis Bach). 
    </div>
</div>


In figure 5, the blue cure is function $f$ which is $\beta$-smooth and $\alpha$-strong convex; it is possible to create global upper and lower quadratic bounds from every point $x \in \mathbb{R}^n$ with respect to curvatures $\beta$ and $\alpha$. 


(Condition number) Let $f$ be a $\beta$-smooth and $\alpha$-strongly convex function on $\mathbb{R}^n$. Then, the condition number of $f$ is defined as

$$
\kappa(f) = \frac{\beta}{\alpha} \tag{21}
$$

For a positive-definite quadratic function f , this definition of the condition number
corresponds with the perhaps more familiar definition of the condition number of the
matrix defining the quadratic.

Now, let's summarize what we have learned into a table. 


|  | Convex | Strongly convex |
|:---|:---:|:---:|
| Lipschitz | $\epsilon \leq \mathcal{O}(1 / t^{1/2})$ | $\epsilon \leq \mathcal{O}(1/t)$ |
| Smooth | $\epsilon \leq \mathcal{O}(1/t)$ | $\epsilon \leq e^{-\alpha t} $ |

Before we move on, let's elaborate on the lower and upper bounds from equation (17) and (19). 

The goal of gradient descent is to find a local minimum of a function $f$. Now, let's assume the optimal value is $x^*$, which means $f(x^*) = \min f(x)$. When we iterate the gradient descent algorithm, we will eventually reach a point $x_t$ that is close to $x^*$. Let's rewrite the inequality (17) as

$$
f(x_t) \leq  f(x^*)  +  \nabla f(x^*)^T (x_t - x^*) + \frac{\beta}{2} \|x_t - x^*\|^2 \tag{22}
$$

and the inequality (19) as

$$
f(x_t) \geq f(x^*)  + \nabla f(x^*)^T (x_t - x^*) + \frac{\alpha}{2} \| x_t - x^* \|^2 \tag{23}
$$

<div class='figure'>
    <img src="/math/images/smooth-and-convex.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The illustration of $\beta$-smooth and $\alpha$-strong convexity again (figure was taken from zhengwenjie). 
    </div>
</div>

The idea is that we are using the parabola approximation of $f$ around $x^*$ to bound the value of $f$ at $x_t$. In practice, having some knowledge of lower and upper bounds on the value of $f$ at $x_t$ is very useful. 

To gain deep understanding of the convergence rate of gradient descent, we need to practice more with concrete examples. In the next section, we will discuss some applications of gradient descent and implement them in Python.


## Some applications of gradient descent

In this section, we will discuss some applications of gradient descent and implement them in Python.


One of the most fundamental data analysis tools is linear least squares. Given an $m \times n$ matrix $A$ and an $m$-dimensional vector $b$, the goal is to find the $n$-dimensional vector $x$ that minimizes the squared error

$$
f(x) = \frac{1}{2m} \|Ax - b\|^2 \tag{24}
$$

We can verify 

$$
\nabla f(x) =  A^T (Ax - b); \quad \nabla^2 f(x) =  A^T A \tag{25}
$$ 

This means $f$ that $f$ is $\beta$-smooth and $\alpha$-strongly convex with 

$$
\beta = \lambda_{\max}(A^T A) \quad \text{and} \quad \alpha =  \lambda_{\min}(A^T A) \tag{26}
$$

Now, we will simulate a dataset with $m = 100$ and $n = 3$ and solve the least squares problem using gradient descent. The following tables 
gives the true coefficients $x$, the initial guess $x_0$, and the final solution $x_t$.


 |   Initial guess (x0) |   True coefficient (x) |   Estimated coefficients (xt) |
|---------------------:|-----------------------:|------------------------------:|
|           -0.831995  |              -1.94281  |                     -1.93834  |
|           -0.0319888 |               1.51628  |                      1.53504  |
|           -0.187633  |              -0.737795 |                     -0.736319 |


<div class='figure'>
    <img src="/math/images/gradient-ols.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 6.</span> The plot of the loss function and its convergence process. 
    </div>
</div>


From the figure 6, we can see that the loss function converges to a minimum after 20 iterations.

The above simulation assumes $m \gg n$. In practice, we often have $m \ll n$ and the least squares problem is ill-posed. Since the matrix is not full rank, the objective function now is no longer strongly convex because 

$$
\lambda_{\min}(A^T A) = 0 \tag{27}
$$

The following table gives the first five true coefficients $x$, the initial guess $x_0$, and the final solution $x_t$.

 |   Initial guess (x0) |   True coefficient (x) |   Estimated coefficients (xt) |
|---------------------:|-----------------------:|------------------------------:|
 |            -2.38525  |             0.00156901 |                     -2.3963   |
|            -0.877133 |            -0.00940305 |                     -0.76494  |
|            -0.166967 |            -0.00913699 |                     -0.1627   |
|            -0.797044 |            -0.00410237 |                     -0.264272 |
|            -0.611308 |            -0.0141191  |                     -0.868437 |


<div class='figure'>
    <img src="/math/images/gradient-ols2.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 7.</span> The plot of the loss function and its convergence process for underdetermined least squares problem.
    </div>
</div>

As it is shown in figure 7, the loss function does not converge to a minimum. This is because the objective function is not strongly convex. In this case, we can use the proximal gradient descent algorithm to solve the problem. The proximal gradient descent algorithm is a variant of gradient descent that adds a proximal term to the objective function.

It is also called the regularized gradient descent algorithm. The proximal term is a regularization term that penalizes the magnitude of the coefficients. The proximal term is defined as

$$
\text{prox}_{g}(x) = \text{argmin}_{y} \left \{ \frac{1}{2} \|y - x\|^2 +g(y) \right \} \tag{28}
$$

Very often, we will focus on the $\ell_1$-norm and the $\ell_2$-norm as the regularization term:

- Ridge penalization: $\ell_2$-norm where $g(y) = \frac{\lambda}{2} \|y\|^2$
- Laplace penalization: $\ell_1$-norm where $g(y) = \lambda \|y\|_1$


Now, for the least squares problem, we can add the ridge penelization term to the objective function. The new objective function is

$$
f(x) = \frac{1}{2m} \|Ax - b\|^2 + \frac{\lambda}{2} \|x\|^2 \tag{29}
$$

|   Initial guess (x0) |   True coefficient (x) |   Estimated coefficients (xt) |
|---:|---------------------:|-----------------------:|------------------------------:|
 |            1.32252   |            6.40826e-05 |                    0.00770425 |
|           -0.980389  |           -0.00379448  |                   -0.0115582  |
|           -0.011439  |            0.00818096  |                    0.00947045 |
|           -0.0894514 |            0.0064178   |                    0.00469251 |
|           -0.131615  |           -0.00768739  |                   -0.00732296 |


<div class='figure'>
    <img src="/math/images/gradient-ols3.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 7.</span> The plot of the loss function and its convergence process for underdetermined least squares problem with ridge penality. 
    </div>
</div>


You see that the error doesn't decrease below a certain level due to the regularization term. This is not a bad thing. In fact, the regularization term gives as strong convexity which leads to convergence in domain again. 


##  Conditional gradient method

Before we move on to the next topic, let's talk about "projection" in general, especially in the context of linear algebra. The dot product of two vectors is a projection of one vector onto the other. If you look at many equations in the above sections, we have done many dot products. For example, this one 

$$
f(y) - f(x) \geq \nabla f(x)^T (y - x)  + \frac{\alpha}{2} \|y - x\|^2 
$$

The dot product $\nabla f(x)^T (y - x)$ is a projection of the gradient of $f$ onto the direction of $y - x$. This is a very important concept in optimization.

The derivative of $f(x)$ not only tells us the direction of the steepest descent, but also the magnitude of the steepest descent. The magnitude of the steepest descent is the projection of the gradient of $f$ onto the direction of $y - x$. 

Again, the optimization problem is to solve the problems of form

$$
\text{minimize}_{x} f(x) 
$$

Therefore, the optimization problem can be summarized as: __finding the minimum of a function by searching $x$ in the domain of $f$.__ But, how do we search $x$ in the domain of $f$? Any problem of searching is a problem of _finding directions_.

Let's state the algorithm of conditional gradient method first, then we will explain it in detail.

Again, we start from the initial guess $x_0$ and we set

$$
x_{t+1} = x_t + \eta_t (\tilde{x}_t - x_t) \tag{30}
$$

where 

$$
\tilde{x}_t = \text{argmin}_{x} \left \{ f(x_t) + \nabla f(x_t)^T(x - x_t) \right \} \tag{31}
$$

Since $f(x_t)$ is pre-determined, we can simplify the above equation to

$$
\tilde{x}_t = \text{argmin}_{x}  \nabla f(x_t)^Tx  \tag{32}
$$

Compared the above equation with the equation of projection:

- $x_{t+1} = x_t - \eta_t \nabla f(x_t)$
- $x_{t+1} = \mathrm{proj}_{\Omega}(x_{t+1})$


The gradient descent in the above updating is to search $x^*$ by following the direction and magnitude of the gradient $\nabla f(x_t)$, whereas the conditional gradient method in equation (30) and (32) is to search $x^*$ by following the projection of the gradient onto the whole domain.

When we are dealing with function in one dimension, the conditional gradient method is equivalent to the gradient descent method. However, when we are dealing with function in higher dimensions, the conditional gradient method is more efficient than the gradient descent method.

Here is the algorithm, starting from $x_0 \in \mathbb{R}^n$, repeat the following steps until convergence:

- Compute $\tilde{x}_t = \text{argmin}_{x}  \nabla f(x_t)^Tx$ (linear search because it is just a dot product)
- Compute $x_{t+1} = x_t + \eta_t (\tilde{x}_t - x_t)$ (update step)

As it turns out, conditional gradient enjoys a convergence guarantee similar to the one
we saw for projected gradient descent.


__Remark:__ we know when two vectors have positive correlation, the dot product is positive. When two vectors have negative correlation, the dot product is negative. When two vectors are orthogonal, the dot product is zero. Therefore, the conditional gradient method is a method that searches $x$ in the direction of the gradient. It is a method that searches $x$ in the direction that correlates most with the steepest descent. Be careful that this is not the same as the steepest descent method. The steepest descent method searches $x$ in the direction of the steepest descent, which is orthogonal to the gradient of $f$.

__It is very important to know that it makes no sense to use the conditional gradient method for the unconstrained optimization problem.__ This is because the linear search in the conditional gradient method needs to know $x$ in the following equation:

$$
\tilde{x}_t = \text{argmin}_{x}  \nabla f(x_t)^Tx , \quad \text{there is no t in } \ x 
$$








{% endkatexmm %}



