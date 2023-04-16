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
- [How to find appropriate initialization values](#how-to-find-appropriate-initialization-values)


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

## Conclusion and code

In this post, we illustrates the importance of the initialization of the weights and the activation functions. We also show that the variance of the gradients and the variance of the activations are preserved across different layers of the neural network with appropriate initialization.

Since the neural network models have become more and more mature, the best practice of the initialization has been well established. The best practice is to use the Xavier initialization for the Tanh and Sigmoid activation functions and the He initialization for the ReLU activation functions. If there will be something new come up in the future, our open source community will be the first one to know about it for sure. So, just stay tuned with the right community.

Here are the code for this post.

```python
# %%
# import packages that are not related to torch
import os
import math
import time
import inspect
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tu_data
from torchvision import transforms
from torchvision.datasets import FashionMNIST


# set up the data path
DATA_PATH = "../data"
SAVE_PATH = "../pretrained/ac_fun"


# function for setting seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# set up seed globally and deterministically
set_seed(76)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# region --------- data prepocessing --------- ###
def _get_data():
    """
    download the dataset from FashionMNIST and transfom it to tensor
    """
    # set up the transformation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )

    # download the dataset
    train_set = FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    return train_set, test_set


def _visualize_data(train_dataset, test_dataset):
    """
    visualize the dataset by randomly sampling
    9 images from the dataset
    """
    # set up the figure
    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3

    for i in range(1, col * row + 1):
        sample_idx = np.random.randint(0, len(train_dataset))
        img, label = train_dataset[sample_idx]
        fig.add_subplot(row, col, i)
        plt.title(train_dataset.classes[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")


# endregion


# region --------- neural network model --------- ###
class BaseNet(nn.Module):
    """
    A simple neural network to show the effect of activation functions
    """

    def __init__(
        self, ac_fun, input_size=784, num_class=10, hidden_sizes=[512, 256, 256, 128]
    ):
        """
        Inputs:
            ac_fun: activation function
            input_size: size of the input = 28*28
            num_class: number of classes = 10
            hidden_sizes: list of hidden layer sizes that specify the layer sizes
        """
        super().__init__()

        # create a list of layers
        layers = []
        layers_sizes = [input_size] + hidden_sizes
        for idx in range(1, len(layers_sizes)):
            layers.append(nn.Linear(layers_sizes[idx - 1], layers_sizes[idx]))
            layers.append(ac_fun)
        # add the last layer
        layers.append(nn.Linear(layers_sizes[-1], num_class))
        # create a sequential neural network
        self.net = nn.Sequential(*layers)  # * is used to unpack the list
        self.num_nets = len(layers_sizes)

        # set up the config dictionary
        self.config = {
            "ac_fun": ac_fun.__class__.__name__,
            "input_size": input_size,
            "num_class": num_class,
            "hidden_sizes": hidden_sizes,
            "num_of_nets": self.num_nets,
        }

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # flatten the input as 1 by 784 vector
        return self.net(x)

    def _layer_summary(self, input_size):
        """
        print the summary of the model
        input_size: the size of the input tensor
                    in the form of (batch_size, channel, height, width)
        note: using * to unpack the tuple
        """
        # generate a random input tensor
        #
        X = torch.rand(*input_size)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)


class Identity(nn.Module):
    """
    A class for identity activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# a dictionary of activation functions
AC_FUN_DICT = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "Identity": Identity}

# endregion


# region --------- visualize weights, gradients and activations --------- ###
def _plot_distribution(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    """
    A helper function for plotting the distribution of the values.
    Input:
        val_dict: a dictionary of values
        color: the color of the plot
        xlabel: the label of the x-axis
        stat: the type of the statistic
        use_kde: whether to use kernel density estimation
    """
    columns = len(val_dict)  # number of columns
    fig, axes = plt.subplots(1, columns, figsize=(columns * 3.5, 3))
    axes = axes.flatten()
    for idx, (name, val) in enumerate(val_dict.items()):
        # only use kde if the range of the values is larger than 1e-7
        sns.histplot(
            val,
            color=color,
            ax=axes[idx],
            bins=50,
            stat=stat,
            kde=use_kde and (val.max() - val.min() > 1e-7),
        )
        axes[idx].set_title(name)
        axes[idx].set_xlabel(xlabel)
        axes[idx].set_ylabel(stat)
    fig.subplots_adjust(wspace=0.4)

    return fig


def visualize_weights(model, color="C0"):
    weights = {}
    # weights are stored in the model as a dictionary called named_parameters
    for name, param in model.named_parameters():
        if "weight" in name:
            weights_key = f"Layer {name.split('.')[1]}"
            weights[weights_key] = param.detach().cpu().numpy().flatten()
    # plot the distribution of the weights
    fig = _plot_distribution(weights, color=color, xlabel="Weight value")
    fig.suptitle("Distribution of the weights", fontsize=14, y=1.05)


def visualize_gradients(model, train_dataset, color="C0", print_variance=False):
    # get the gradients by passing a batch of data through the model
    model.eval()
    data_loader = tu_data.DataLoader(train_dataset, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(data_loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    # calculate the gradients
    model.zero_grad()
    preds = model(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    gradients = {
        name: param.grad.detach().cpu().numpy().flatten()
        for name, param in model.named_parameters()
        if "weight" in name
    }
    model.zero_grad()

    # plot the distribution of the gradients
    fig = _plot_distribution(gradients, color=color, xlabel="Gradient value")
    fig.suptitle("Distribution of the gradients", fontsize=14, y=1.05)

    if print_variance:
        for name, grad in gradients.items():
            print(f"{name} gradient variance: {np.var(grad):.2e}")


def visualize_activations(model, train_dataset, color="C0", print_variance=False):
    model.eval()
    data_loader = tu_data.DataLoader(train_dataset, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(data_loader))
    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

    # pass the data through the model
    img_features = imgs.view(imgs.shape[0], -1)
    activations = {}

    with torch.no_grad():
        for idx, layer in enumerate(model.net):
            img_features = layer(img_features)
            if isinstance(layer, nn.Linear):
                activations[f"Layer {idx}"] = (
                    img_features.detach().cpu().numpy().flatten()
                )
    # plot the distribution of the activations
    fig = _plot_distribution(
        activations, color=color, stat="density", xlabel="Activation value"
    )

    fig.suptitle("Distribution of the activations", fontsize=14, y=1.05)

    if print_variance:
        for name, act in activations.items():
            print(f"{name} activation variance: {np.var(act):.3f}")


# endregion


# constant initialization
def const_init(model, val=0.0):
    for name, param in model.named_parameters():
        param.data.fill_(val)


def const_variance_init(model, std=0.01):
    for name, param in model.named_parameters():
        param.data.normal_(0, std)


def xavier_init(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
        elif "bias" in name:
            param.data.fill_(0.0)


def kaiming_init(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0.0)
        elif name.startswith("layers.0"):
            # the first layer does not have ReLU
            param.data.normal_(0, 1 / np.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, np.sqrt(2) / np.sqrt(param.shape[1]))


if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", DEVICE)
    set_seed(42)
    # set up finger config
    # it is only used for interactive mode
    # %config InlineBackend.figure_format = "retina"

    # download the dataset
    train_set, test_set = _get_data()
    print("Train set size:", len(train_set))
    print("Test set size:", len(test_set))
    print("The shape of the image:", train_set[0][0].shape)
    print("The size of vectorized image:", train_set[0][0].view(-1).shape)
    # calculate the mean of the image
    print("check features", train_set.data.shape)
    print("check labels", train_set.targets.shape)
    # calculate the mean of the image by dividing 255
    # 255 is the maximum value of the image (0 - 255 pixel value)
    print("The mean of the image:", train_set.data.float().mean() / 255)
    print("The standard deviation of the image:", train_set.data.float().std() / 255)

    # load the dataset
    train_loader = tu_data.DataLoader(
        train_set, batch_size=1024, shuffle=True, drop_last=False
    )
    imgs, _ = next(iter(train_loader))
    print(f"Mean: {imgs.mean().item():5.3f}")
    print(f"Standard deviation: {imgs.std().item():5.3f}")
    print(f"Maximum: {imgs.max().item():5.3f}")
    print(f"Minimum: {imgs.min().item():5.3f}")
    print(f"Shape of tensor: {imgs.shape}")

    our_model = BaseNet(ac_fun=Identity()).to(DEVICE)

    # ----  constant initialization
    const_init(our_model, val=0.005)
    visualize_gradients(our_model, train_set)
    visualize_activations(our_model, train_set, print_variance=True)
    print("-" * 80)

    # ---- constant variance initialization
    const_variance_init(our_model, std=0.01)
    visualize_activations(our_model, train_set, print_variance=True)
    print("-" * 80)

    const_variance_init(our_model, std=0.1)
    visualize_activations(our_model, train_set, print_variance=True)

    # ---- xavier initialization
    xavier_init(our_model)
    visualize_weights(our_model)
    visualize_activations(our_model, train_set, print_variance=True)

    # ---- kaiming initialization
    our_model = BaseNet(ac_fun=nn.ReLU()).to(DEVICE)
    kaiming_init(our_model)
    visualize_gradients(our_model, train_set, print_variance=True)
    visualize_activations(our_model, train_set, print_variance=True)
``` 














{% endkatexmm %}


## References

1. [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html)
2. [University of Amsterdam Notes](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html)
3. [How to initialize deep neural networks? Xavier and Kaiming initialization](https://pouannes.github.io/blog/initialization/#mjx-eqn-eqfwd_K)



