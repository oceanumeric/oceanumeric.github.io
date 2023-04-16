---
title: Deep Learning Parameters Initialization 
subtitle: Training and tuning a deep learning model is a complex process. This post will cover the basics of how to initialize the parameters of a deep learning model.
layout: blog_default
date: 2023-04-16
keywords: deep learning, pytorch, data science, data collection, data processing, data analysis, model training
published: true
tags: deep-learning pytorch data-science python
---

In previous posts, I have explained why we should treat deep learning as an architecture construction problem, which means one should think of the neural network as an architecture that can be built from a set of components: 

- the material used to build the architecture (the data)
- the infrastructure used to build the architecture (the hardware)
- the tools used to build the architecture (the software)
- the design of the architecture (the model) 

In this post, I will focus on the design of the architecture, which is the model. The model needs to be trained and tuned to achieve the best performance with many parameters to be set. This post will cover the basics of how to initialize the parameters of a deep learning model.

- [Big idea](#big-idea)
- [Data distribution](#data-distribution)
- [Constant initialization](#constant-initialization)
- [Constant variance initialization](#constant-variance-initialization)


## Big idea 

{% katexmm %}

Deep learning models are about mapping a set of inputs to a set of outputs. The best model is to link inputs to outputs with the smallest error. The error is the difference between the predicted output and the actual output.

From the perspective of _information theory_, the best model is the one that minimizes the _cross entropy_ between the predicted output and the actual output. The cross entropy is a measure of the difference between two probability distributions. The cross entropy is defined as:

$$H(p,q) = -\sum_{i=1}^n p_i \log q_i \tag{1} $$ 

where $p$ is the actual probability distribution and $q$ is the predicted probability distribution. The cross entropy is a measure of the difference between the actual distribution and the predicted distribution. The smaller the cross entropy, the better the model. To be honest, I don't fully understand equation (1). I might write a post about it in the future. But, right now, we could still understand the importance of initialization and optimization from the perspective of information theory without digging into equation (1). 


Intuitively, _we want to extract the most information from the data and preserve the most information across different layers of neural networks_. The more information we can extract from the data, the better the model. The more information we can preserve across different layers of neural networks, the better the model.

There is a very good book discussing neural networks from the perspective of information theory: _Information Theory, Inference, and Learning Algorithms_ by David MacKay {% cite mackay2003information %}. I highly recommend this book to anyone who wants to understand neural networks from the perspective of information theory.


## Data distribution

There are many ways to quantify the information in a dataset. The simplest way is to use the distribution of the data. In this post, we will focus on the first two moments of the data distribution: the mean and the variance.

The idea is that _we want to initialize the parameters of the model such that the mean and the variance of the data are preserved across different layers of the neural network_. When we talk about the preservation of the mean and the variance, we are more interested in
the overall shape of the distribution because there is no way to preserve the exact mean and the exact variance with many layers of neural networks (there are so many transformations involved in a neural network).

We will use Fashion-MNIST dataset to illustrate the idea as we did it in [previous posts](https://oceanumeric.github.io/blog/activation-functions){:"target=_blank"}.

The following comments shows the mean and the variance of the data distribution before and after normalization. The purpose of normalization is to make the mean of the data distribution to be zero and the variance of the data distribution to be one. Then, it will be easier to compare the mean and the variance of the data distribution across different layers of the neural network.

```python
# code to produce the following comments will be given at the end of this post
# Train set size: 60000
# Test set size: 10000
# The shape of the image: torch.Size([1, 28, 28])
# The size of vectorized image: torch.Size([784])
# check features torch.Size([60000, 28, 28])
# check labels torch.Size([60000])
# The mean of the image: tensor(0.2860)
# The standard deviation of the image: tensor(0.3530)

# ------------- after normalization -------------
# Mean: -0.006
# Standard deviation: 0.999
# Maximum: 2.023
# Minimum: -0.810
# Shape of tensor: torch.Size([1024, 1, 28, 28])
```
If you look at the last line of the comments, you will see the tensor is four-dimensional. PyTorch convert the images into one-dimensional vectors for calculating mean and standard deviation. We will not cover how PyTorch does it in this post. But, it is important to be aware of dimensions of our data. 

To learn how initialization will affect the mean and variance of the data distribution, gradient descent and activation functions, we will define several functions to visualize the data distribution. Those functions will help us:

- visualize the weight/parameter distribution
- visualize the gradient distribution
- visualize the data distribution after activation function (propagation)


## Constant initialization

When we train a neural network with PyTorch, PyTorch will initialize the parameters of the model automatically. We can also initialize the parameters of the model manually. In this section, we will initialize the parameters of the model with a constant value. Then, we will visualize the data distribution after the propagation to see how the mean and the variance of the data distribution are affected by the constant initialization.

<div class='figure'>
    <img src="/images/blog/initialization-gradients-1.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The visualization of the gradients with constant initialization = 0.005 (you can click on the image to zoom in).
    </div>
</div>

As we can see, only the first and the last layer have diverse gradient distributions while the other three layers have the same gradient for all weights (note that this value is unequal 0, but often very close to it). Having the same gradient for parameters that have been initialized with the same values means that we will always have the same value for those parameters. This would make our layer useless and reduce our effective number of parameters to 1 (meaning that we could have just used a single parameter instead of many).


<div class='figure'>
    <img src="/images/blog/initialization-activation-1.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The visualization of activations with constant initialization = 0.005 (you can click on the image to zoom in).
    </div>
</div>

Figure 2 gives the visualization of the data distribution after the propagation. We can see that shape was preserved across different layers of the neural network. The following comments show the variance of the data distribution across different layers of the neural network. The variance of the data distribution is a measure of the spread of the data distribution. The smaller the variance, the more information we can extract from the data. The larger the variance, the less information we can extract from the data. _We do not want to our model to have a growing variance across different layers of the neural network because it will reduce the information we can extract from the data_ (meaning the model explodes).


```python
# Layer 0 activation variance: 1.941
# Layer 2 activation variance: 12.720
# Layer 4 activation variance: 20.841
# Layer 6 activation variance: 34.145
# Layer 8 activation variance: 13.986
```

## Constant variance initialization

In this section, we will initialize the parameters of the model with a constant variance. Then, we will visualize the data distribution after the propagation to see how the mean and the variance of the data distribution are affected by the constant variance initialization.

First, we will initialize the parameters of the model with a constant variance by setting $std=0.01$. Second, we will initialize the parameters of the model with a constant variance by setting $std=0.1$.

<div class='figure'>
    <img src="/images/blog/initialization-activation-2.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The visualization of activations with constant variance initialization = 0.01 (you can click on the image to zoom in).
    </div>
</div>

```python
# Layer 0 activation variance: 0.076
# Layer 2 activation variance: 0.004
# Layer 4 activation variance: 0.000
# Layer 6 activation variance: 0.000
# Layer 8 activation variance: 0.000
```

Figure 3 and the above comments show that the variance vanishes across different layers of the neural network. This means that the model is not learning anything. The model is not learning anything because the variance of the data distribution is too small. However, when we set $std=0.1$, the variance of the data distribution explodes across different layers of the neural network. This means that the model is learning too fast and it is not able to learn anything either as it is shown in the following figure and comments.

<div class='figure'>
    <img src="/images/blog/initialization-activation-3.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The visualization of activations with constant variance initialization = 0.1 (you can click on the image to zoom in).
    </div>
</div>

```python
# Layer 0 activation variance: 7.919
# Layer 2 activation variance: 41.948
# Layer 4 activation variance: 106.103
# Layer 6 activation variance: 289.248
# Layer 8 activation variance: 352.923
```

## How to find appropriate initialization values

To prevent the gradients of the network’s activations from vanishing or exploding, we will stick to the following rules of thumb:

1. The mean of the activations should be zero.
2. The variance of the activations should stay the same across every layer.

This means we will preserve the 'information structure' of the data across different layers of the neural network.

Suppose we want to come up the initialization for the following layer:

$$
y = Wx + b, \quad W \in \mathbb{R}^{n \times m}, \quad x \in \mathbb{R}^{m \times 1}, \quad b \in \mathbb{R}^{n \times 1}, \quad y \in \mathbb{R}^{n \times 1}
$$

Our goal is to find the appropriate initialization for $W$ and $b$ and make sure variance of the activations is preserved across different layers of the neural network. We will use the following notation:

$$
\mathrm{Var}(y) = \mathrm{Var}(x) = \sigma^2_x \tag{2}
$$

Now, we will derive the appropriate initialization for $W$. We let $b$ be zero for simplicity because the bias term does not affect the variance of the activations (it only affects the mean of the activations based on the linearity of the expectation operator). For those who want to refresh their memory about the variance of the linear transformation, please refer to one of [my post](https://oceanumeric.github.io/math/2023/01/probability-review){:target="_blank"}.

We have:

$$
y = \sum_{j} w_j x_j, \quad \text{where j is the index of the input dimension}
$$

With the condition set in equation (2), we have:

$$
\begin{aligned}
\mathrm{Var}(y) = \sigma^2_x & = \mathrm{Var}(\sum_{j} w_j x_j) \\
& = \sum_{j} \mathrm{Var}(w_j x_j) \\   
& = \sum_{j} \mathrm{Var}(w_j) \mathrm{Var}(x_j) \quad \text{because } \mathrm{Cov}(w_j, x_j) = 0 \\
& = \sigma_x^2 \cdot d_x \cdot \mathrm{Var}(w_j) \quad \text{d is the number of input dimensions} \\ 
\Rrightarrow \mathrm{Var}(w_j) & = \frac{1}{d_x} \tag{3}
\end{aligned}
$$

Many steps of derivation in the above equation depends on the independence of the weight and the input.

Equation (3) shows that we should initialize the weight distribution with a variance of the inverse of the input dimension $d_x$. This kind of initialization is called Xavier initialization {% cite glorot2010understanding %}. Figure 5 shows the visualization of the weights with Xavier initialization. Since we initialize all weights with the normal distribution, the mean of the weights is zero. The variance of the weights is the inverse of the input dimension for each layer.


<div class='figure'>
    <img src="/images/blog/initialization-weights-1.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The visualization of weights with Xavier initialization (you can click on the image to zoom in).
    </div>
</div>

Now, let's check whether the variance of the activations is preserved across different layers of the neural network. 

<div class='figure'>
    <img src="/images/blog/initialization-activation-4.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The visualization of activations with Xavier initialization (you can click on the image to zoom in).
    </div>
</div>

```python
# 0.964 is the variance of the data distribution
# Layer 0 activation variance: 0.964
# Layer 2 activation variance: 1.002
# Layer 4 activation variance: 0.985
# Layer 6 activation variance: 0.938
# Layer 8 activation variance: 1.225
```

Figure 6 and the above comments show that the variance of the activations is preserved across different layers of the neural network. This means that the model is learning something.

There are different kinds of variants of the Xavier initialization. If you want to learn more about the Xavier initialization, you can google it. But I think it is not necessary to know all the variants of the Xavier initialization. The most important thing is to know that we should initialize the weights with a variance of the inverse of the input dimension. 

As I have pointed it out that building a neural network is an art of architecture design. _The best way to learn the state of the art architecture design is to follow the open source community as there are many talented people who are willing to share their knowledge with the community about the state of the art architecture design_.


### He initialization

The Xavier initialization works well for Tanh and Sigmoid activation functions. However, it does not work well for ReLU activation functions. For the ReLu activation function, we should use the He initialization {% cite he2015delving %}. The He initialization is the same as the Xavier initialization except that we use the variance of the inverse of the input dimension multiplied by 2 such as 

$$
\mathrm{Var}(w_j) = \frac{2}{d_x} \tag{4}
$$


<div class='figure'>
    <img src="/images/blog/initialization-gradients-2.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 6.</span> The visualization of the gradients with He initialization (you can click on the image to zoom in).
    </div>
</div>

You can see that the gradients are much diverse with the He initialization compared to the Figure 1 where we use the constant initialization. This means that the model is learning something. The variance of gradients is also quite stable.

```python
# variance of gradients
# net.0.weight gradient variance: 6.67e-05
# net.2.weight gradient variance: 1.68e-04
# net.4.weight gradient variance: 2.36e-04
# net.6.weight gradient variance: 5.38e-04
# net.8.weight gradient variance: 7.11e-03
```

For the activation function, Figure 7 shows the visualization of the activation with the He initialization. You can see that the variance of the activation is preserved across different layers of the neural network.

<div class='figure'>
    <img src="/images/blog/initialization-activation-5.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 7.</span> The visualization of activations with he initialization (you can click on the image to zoom in).
    </div>
</div>

```python
# each layer activation variance doubled compared to the Xavier initialization
# Layer 0 activation variance: 1.928
# Layer 2 activation variance: 2.068
# Layer 4 activation variance: 1.968
# Layer 6 activation variance: 1.817
# Layer 8 activation variance: 1.622
```














{% endkatexmm %}


## References

1. [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
2. [University of Amsterdam Notes](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html)



