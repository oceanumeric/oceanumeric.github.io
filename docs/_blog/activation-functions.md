---
title: Understanding Activation Functions in Neural Networks
subtitle: As key components of neural networks, activation functions are responsible for transforming the input data into the desired output. In this article, we will discuss the most common activation functions and their applications.
layout: blog_default
date: 2023-04-14
keywords: deep learning, pytorch, data science, data collection, data processing, data analysis, model training
published: true
tags: deep-learning pytorch data-science python
---


After the innovation like ChatGPT, I begun to think about the fundamentals of deep learning models. In the old days (I mean three to five years ago), I had some doubt on deep learning models as there is lack of theoretical support. Compared to traditional machine learning and statistcal models, deep learning models are more like black boxes. However, somehow it works and it is creating a lot of impact in the industry.

Therefore, I updated my perception on deep learning models. Instread of caring a lot about the theoretical support, I think it is more important to understand the mechanism of deep learning models. It is better to think about the deep learning models as a nerual network architecture that could map data from one space to another space. 

With this perspective, one could understand how two architectures have changed our world. One is the word2vec model, which maps words from one space to another space. The other is the transformer model, which maps sentences from one space to another space. As the famous paper stated, the attention is all we need. 

In this article, I will talk about the activation functions. Since activation functions are the core of deep learning models, I think it is important to understand the mechanism of activation functions.


## Automatic differentiation

{% katexmm %}

Before we talk about activation functions, let's talk about the automatic differentiation. In the old days, we need to manually calculate the gradient of the loss function with respect to the parameters. However, with the automatic differentiation, we could use the backpropagation algorithm to calculate the gradient of the loss function with respect to the parameters.

The automatic differentiation is the key to the success of deep learning models. With the automatic differentiation, we could train the deep learning models with the gradient descent algorithm. If you learn the concept of automatic differentiation in one framework, you could easily apply it to other frameworks. In this post, I will use the PyTorch framework to demonstrate the concept of automatic differentiation.

Let's say we have a vector $x$, and the following two
functions:

$$
x  = \begin{bmatrix} 0 \\ 1 \\ 2 \\ 3 \end{bmatrix}, \quad
y_1 = x^T x  = [14], \quad
y_2  = x \odot x = \begin{bmatrix}
0 \\ 1 \\ 4 \\ 9
\end{bmatrix}  \tag{1}
$$

The derivative of $y$ with respect to $x$ is:

$$
\frac{\partial y_1}{\partial x} = 2x  = \begin{bmatrix} 0 \\ 2 \\ 4 \\ 6  \end{bmatrix}, \quad 
\frac{\partial y_2}{\partial x} = 2x = \begin{bmatrix} 0 \\ 4 \\ 4 \\ 6  \end{bmatrix} \tag{2}
$$

Although the two functions have the same derivative, `PyTorch` calculate the derivative in a different way as the first one is a scalar and the second one is a vector. The other thing to remember is that the `PyTorch` will accumulate the gradient of the loss function with respect to the parameters. Therefore, we need to set the gradient to zero before we calculate the gradient of the loss function with respect to the parameters. 

During the training process, PyTorch will construct a graph to record the operations. The graph is called the computational graph. The computational graph is used to calculate the gradient of the loss function with respect to the parameters. The following figure illustrates the computational graph.

<div class='figure'>
    <img src="/images/blog/backward-grad-fig-1.png"
         alt="Pytorch autograd illustration"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The computational graph.
    </div>
</div>

With equations (1) and (2), we will show how to calculate the gradient in PyTorch. Three points need to be mentioned:

1. The `requires_grad_()` is set to `True` to record the operations in the computational graph.
2. The gradients vary with the input data $x$ and function $y$, therefore,
    - `.grad` is an attribute of the tensor.
    - `.grad` is None if the tensor does not require gradient.
    - `.grad` is accumulated if the tensor requires gradient.
    - `.grad` will be updated after the `backward()` function is called.
    - `.grad.zero_()` is used to set the gradient to zero if you want to calculate the gradient of a different loss function with respect to the same parameters manually.
3. The `backward()` function is used to calculate the gradient of the loss function with respect to the parameters.

Here is the code will produce a `RuntimeError`:

```python
x = torch.arange(4.0)  # initialize tensor
print(x.grad) # no gradient
# set gradient attribute
x.requires_grad_(True)  # default is 
# equivalent to x.requires_grad = True
# or x = torch.arange(4.0, requires_grad=True)
print(x.grad) # still None
y1 = torch.dot(x, x)
print(y1)
y1.backward()
print(x.grad)
y2 = x * x
print(y2)
y2.backward()  # grad can be implicitly created only for scalar outputs
```

Here is the code will produce the correct result:

```python
x = torch.arange(4.0, requires_grad=True)
print(x)
# tensor([0., 1., 2., 3.], requires_grad=True)
y1 = torch.dot(x, x)
print(y1)
# tensor(14., grad_fn=<DotBackward0>)
y1.backward()
print(x.grad)
# tensor([0., 2., 4., 6.])
print("Without zeroing the gradient")
y2 = x * x
y2.sum().backward()
print(x.grad)
# tensor([ 0.,  4.,  8., 12.])
# you can see that the gradient is accumulated
print("Setting the gradient to zero")
x.grad.zero_()
print(x.grad)
# tensor([0., 0., 0., 0.])
y2 = x * x
print(y2)
# tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
y2.backward(torch.ones_like(x))
print(x.grad)
# tensor([0., 2., 4., 6.])
```

_Remark:_ since the gradient does not depend on the value of the output $y$, that's why the following two lines of code produce the same result:

```python
# when the function is related to vector operations
y2.backward(torch.ones_like(x))
y2.sum().backward()
```

One last thing you need to know that PyTorch has a function called `detach()` which will detach the tensor from the computational graph. Doing so could give us the gradient of the loss function with respect to the certain parameters in certain layers or steps.

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad == u  # tensor([True, True, True, True])
# refernce https://d2l.ai/chapter_preliminaries/autograd.html#detaching-computation
``` 

## Activation functions


The activation functions are used to introduce non-linearity to the neural network. The following figure illustrates the activation functions.

<div class='figure'>
    <img src="/images/blog/activation-funs-1.png"
         alt="Pytorch autograd illustration"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The visualization of the activation functions.
    </div>
</div>

According to the figure 1, different activation functions give different non-linear behavior. The following figure illustrates the non-linear behavior of the activation functions. However, the key point is that the effect will be triggered when the input is large enough. And the gradient of the activation function will capture this effect.


## Analyzing the effect of activation functions

After we have a basic understanding of the activation functions, we will analyze the effect of the activation functions. We will train a neural network with different activation functions and compare the results. The dataset we will use is the Fashion-MNIST dataset.


When we choose the activation function, we want to make sure that the
gradient will not vanish or explode. Since PyTorch has a built-in function, the chance of the gradient exploding is low. However, the gradient vanishing is still a problem. The following figure illustrates the gradient vanishing problem.

<div class='figure'>
    <img src="/images/blog/activation-funs-2.png"
         alt="Pytorch autograd illustration" class = "zoom-img"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The histogram of the gradient for different activation functions (you can zoom in to see the details).
    </div>
</div>



















{% endkatexmm %}



## References

1. [https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/](https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/){:target="_blank"}