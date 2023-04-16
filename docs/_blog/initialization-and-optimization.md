---
title: Deep Learning Parameters Initialization and Optimization
subtitle: Training and tuning a deep learning model is a complex process. This post will cover the basics of how to initialize and optimize the parameters of a deep learning model.
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

In this post, I will focus on the design of the architecture, which is the model. The model needs to be trained and tuned to achieve the best performance with many parameters to be set. This post will cover the basics of how to initialize and optimize the parameters of a deep learning model.

- [Big idea](#big-idea)
- [Data distribution](#data-distribution)


## Big idea 

{% katexmm %}

Deep learning models are about mapping a set of inputs to a set of outputs. The best model is to link inputs to outputs with the smallest error. The error is the difference between the predicted output and the actual output.

From the perspective of _information theory_, the best model is the one that minimizes the _cross entropy_ between the predicted output and the actual output. The cross entropy is a measure of the difference between two probability distributions. The cross entropy is defined as:

$$H(p,q) = -\sum_{i=1}^n p_i \log q_i$$ \tag{1}

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








{% endkatexmm %}




