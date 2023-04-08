---
title: Full Stack Deep Learning with PyTorch
subtitle: A guide to building a full stack deep learning application with PyTorch in a small scale, from data collection to model saving without deploying to production.
layout: blog_default
date: 2023-04-08
keywords: deep learning, pytorch, data science, data collection, data processing, data analysis, model training
published: true
tags: deep-learning pytorch data-science python
---


Back in 2017, I have to use `numpy` to train a neural network. It was a pain. One has to write a lot of code 
to train a simple neural network. The chance of having a bug is high. Later, I went to study math and statistics. I did not follow the deep learning trend. I thought it would be better to learn the math and statistics first, which should be the foundation of deep learning, such as optimization, probability, and statistics.

I gained a lot of knowledge from the math and statistics courses. This year, I started to train neural networks again. I found that the deep learning frameworks have become much more mature. I am very impressed by the `pytorch` framework. 

In this post, I will show you how to train a neural network with `pytorch`. I will use the `pytorch` framework to train a neural network to classify the fashion-MNIST dataset. The fashion-MNIST dataset is a dataset of Zalando's article images. It is a drop-in replacement for the MNIST dataset. It has 10 classes, such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. The dataset has 60,000 training examples and 10,000 testing examples. 


## Roadmap

We will grow the code base step by step. The code base will be organized in a modular way. We will use the test driven development (TDD) method to develop the code.

The style of the post is more like a tutorial step by step. It not only shows you how to train a neural network with `pytorch`, but also shows you how to build up a code base that is easy to debug and easy to maintain and extend.

- [The complexity of training a neural network](#the-complexity-of-training-a-neural-network)
- [Managing the dataset](#managing-the-dataset)
- [Test driven development](#test-driven-development)
- [The data preprocessing](#the-data-preprocessing)
- [Build a model](#build-a-model)
- [Train the model](#train-the-model)


## The complexity of training a neural network

The training of a neural network is a complex process. It involves many steps. It also involves many hyperparameters turning. The hyperparameters include the learning rate, the number of epochs, the batch size, the optimizer, the loss function, the activation function, the initialization method, and so on. 

All those hyperparameters are important. They can affect the performance of the neural network. Therefore, training a neural network is a time-consuming process. It is becoming more or less an art, which is highly of practical skills.

Besides, there are many steps in the training of a neural network. It involves the data preprocessing, the data loading, the model building, the model training, the model evaluation, and the model saving.

Therefore, it is important to build up a code base that is easy to debug and easy to maintain. It is also important to have a structure that is easy to extend.


Since modules and functions of  `pytorch` itself have grown a lot, it
is better to import packages separately. The following code imports
the packages that we will use in this post. It also sets up the device 
and other parameters.

```python
# import packages that are not related to torch
import os
import math
import time
import numpy as np
import seaborn as sns 
from tqdm import tqdm
import matplotlib.pyplot as plt


# torch import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tu_data
from torchvision.datasets import FashionMNIST


### --------- environment setup --------- ###
# set up the data path
DATA_PATH = "../data"

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
``` 

## Managing the dataset

The `pytorch` framework provides a lot of datasets. We can use the `torchvision.datasets` package to manage the datasets. The following code downloads the fashion-MNIST dataset and transforms it into a tensor. The `transforms` package provides a lot of transformation functions. We can use the `transforms.Compose` function to compose a transformation. The `transforms.Normalize` function normalizes the data. The `transforms.ToTensor` function transforms the data into a tensor.

```python
# set up the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# download the dataset
train_set = FashionMNIST(root='./data', train=True,
                                download=True, transform=transform)
test_set = FashionMNIST(root='./data', train=False, download=True)
``` 

If you collect and organize the data by yourself, you can use the `torch.utils.data.TensorDataset` class to create a dataset. We will talk about this in the second part of this post.

## Test driven development

The `pytorch` framework provides a lot of modules and functions. It is easy to make mistakes. Therefore, we should use the test driven development (TDD) method to develop the code. The TDD method is a software development process that relies on the repetition of a very short development cycle: requirements are turned into very specific test cases, then the software is improved to pass the new tests, only. 

The key idea of the TDD method is to write a test before writing the code. However, since we are training a neural network, it does not make sense to write so many tests before writing the code. Furthermore, the training of a neural network is different from developing a software, especially at the early stage. Much of the work involves _trial and error_. 

However, we still want to build up our code base in a testable way.
How could we do this? I think the best way is to use `functions` and `classes` to build up the code base. The `functions` and `classes` are the basic building blocks of the `pytorch` framework. We can use them to build up the code base. 

Once you have different `functions` and `classes`, we can call them in the manner of a pipeline. This means we can debug each `function` and `class` separately in the pipeline.

```python
def fun1():
    pass

Class Class1():
    def __init__(self):
        pass

    def fun2(self):
        pass


if __name__ == "__main__":
    # build up the pipeline
    fun1()
    class1 = Class1()
    class1.fun2()
    # debug each function and class separately
``` 

## The data preprocessing

The data preprocessing is the first step in the training of a neural network. It involves the data loading, the data splitting, and the data normalization.

Here is the code at this stage. 

```python
# %% 
# import packages that are not related to torch
import os
import math
import time
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


### --------- environment setup --------- ###
# set up the data path
DATA_PATH = "../data"

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --------- data preprocessing --------- ###

def _get_data():
    """
    download the dataset from FashionMNIST and transfom it to tensor
    """
    # set up the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # download the dataset
    train_set = FashionMNIST(root='./data', train=True,
                                    download=True, transform=transform)
    test_set = FashionMNIST(root='./data', train=False,
                                    download=True, transform=transform)
    
    return train_set, test_set



if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", device)
    
    # download the dataset
    train_set, test_set = _get_data()
```

You can run above code in the interactive window in VS Code or in command line. It is important to know:

- anything is wrapped in the function will be only executed when you call the function.
- anything is not wrapped in the function will be executed when you run the script. For instance, `set_seed(76)` will be executed when you run the script.

What are advantages of using the TDD-oriented code? This allows us:

- to debug each function and class separately.
- to build up the code base in a testable way.
- to set up global variables in a testable way.
- to organize the code in a pipeline way.
- to have a clear idea of the data flow and structure that allows us to extend and scale the code base.

There are two main `attributes` of the dataset: `train_set.data` and `train_set.targets`. The `train_set.data` is a tensor with shape `(60000, 28, 28)`. The `train_set.targets` is a tensor with shape `(60000,)`. The `train_set.data` is the input data. The `train_set.targets` is the label data.

```python
# understand the dataset
print("The size of the training set is", len(train_set))
print("The dimension of the training set is", train_set.data.shape)
print("The dimension of the training label is", train_set.targets.shape)
print("The labels are", train_set.classes)
print("The labels with index are", train_set.class_to_idx)

# The size of the training set is 60000
# The dimension of the training set is torch.Size([60000, 28, 28])
# The dimension of the training label is torch.Size([60000])
```

The attributes of the training dataset in `pytorch` are quite important and useful. In the sense that we can learn how to build up the training dataset for our own data.

Since many of the attributes are not intuitive, we can use the `dir` function to check the attributes of the dataset.

```python
# check the attributes of the dataset
print(dir(train_set))
```

One of the attribute is `__getitem__`. This attribute allows us to access the data and label by index. 

```python
# access the data and label by index
feature, label = train_set[0]
```

The easiest way to learn more about the attributes of the dataset is to check the source code of the dataset [here](https://pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#FashionMNIST){:target="_blank"}. 


When we train a neural network, we need to split the dataset into training set and validation set. The validation set is used to evaluate the performance of the model during the training process. The validation set is not used to train the model. 

```python
# split the dataset into training set and validation set
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = tu_data.random_split(train_set, [train_size, val_size])
```

## Build a model

The model is the core of the neural network. The model is a class that inherits from the `nn.Module` class. The `nn.Module` class is the base class for all neural network modules. The `nn.Module` class provides a lot of useful attributes and methods. 

A neural network module has two main methods: `__init__` and `forward`. The `__init__` method is used to initialize the parameters of the model. The `forward` method is used to define the forward pass of the model. 

Building a model is a very important step in the deep learning pipeline. The model is the core of the neural network. When you have many layers of the neural network, it is very important to organize the model in a modular way. __it is also very important to track the dimension of the data in each layer__. To do this, we need a helper function `_layer_summary` to print out the dimension of the data in each layer. 

The following code shows how I build up a neural network model based on LeNet-5 architecture at this stage. The model has not been fully built up yet. I just want to show you the process of building up a model. 

```python
# %% 
# import packages that are not related to torch
import os
import math
import time
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


### --------- environment setup --------- ###
# set up the data path
DATA_PATH = "../data"

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --------- data preprocessing --------- ###

def _get_data():
    """
    download the dataset from FashionMNIST and transfom it to tensor
    """
    # set up the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # download the dataset
    train_dataset = FashionMNIST(root='./data', train=True,
                                    download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False,
                                    download=True, transform=transform)
    
    return train_dataset, test_dataset


def _visualize_data(train_dataset, test_dataset):
    """
    visualize the dataset by randomly sampling
    9 images from the dataset
    """
    # set up the figure
    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3
    
    for i in range(1, col*row+1):
        sample_idx = np.random.randint(0, len(train_dataset))
        img, label = train_dataset[sample_idx]
        fig.add_subplot(row, col, i)
        plt.title(train_dataset.classes[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        

# build the model
class FashionClassifier(nn.Module):

    def __init__(self):
        # initialize the parent class
        # super() allows us to access methods from a parent class
        super(FashionClassifier, self).__init__()
        # define the neural network
        # based on architecture of LeNet-5
        self.net = nn.Sequential(
            # convolutional layers
            # input channel 1 (grayscale), output channel 6
            # kernel size 5 and stride is default 1
            # kernel is just a fancy name for filter
            # here we are using 5 x 5 filter
            # we are using padding to keep the output size the same
            # the output size is (28 - 5 + 2 * 2) / 1 + 1 = 28
            nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5,
                                     padding=2),
            # activation function sigmoid
            nn.Sigmoid(),
            # average pooling layer
            # the output size is (28 - 2) / 2 + 1 = 14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # the output size is (14 - 5) / 1 + 1 = 10
            nn.Conv2d(in_channels=6, out_channels=16,
                                     kernel_size=5),
            nn.Sigmoid(),
            # the output size is (10 - 2) / 2 + 1 = 5
            nn.AvgPool2d(kernel_size=2, stride=2),
            # dense layers or fully connected layers
            nn.Flatten()
        )
        
        
    def _layer_summary(self, input_size):
        """
        print the summary of the model
        Input:
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
        
    

# function for training the model
def train_the_model(training_dataset, val_dataset):
    
    # create the loader
    training_loader = tu_data.DataLoader(training_dataset,
                                            batch_size=64,
                                            shuffle=True)
    validation_loader = tu_data.DataLoader(val_dataset,
                                            batch_size=64,
                                            shuffle=True)
    
def main():
    # download the dataset
    train_dataset, test_dataset = _get_data()
    _visualize_data(train_dataset, test_dataset)
    
    # split the dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = tu_data.random_split(train_dataset,
                                                      [train_size, val_size])

    

if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", device)

    # check layer summary
    foo_model = FashionClassifier()
    foo_model._layer_summary((1, 1, 28, 28))
    
```

You can notice that I added a `main()` function which is slowly building up my pipeline. At the same time, I am using the code after the `if __name__ == "__main__":` to test the code. The `main()` function will be called at the end of the script.

I think now you can appreciate the power of test driven development.
For instance, in the above code, I am testing a new function called `_layer_summary()` which will print the summary of each layer in the neural network and help me to trace the dimension of the tensor.

Anyone who has some experience in building a neural network will know that the dimension of the tensor is very important. You know what I am talking about if you have ever encountered the error message `RuntimeError: Given groups=1, weight of size [6, 1, 5, 5], expected input[64, 6, 28, 28] to have 1 channels, but got 6 channels instead`.



## Train the model


Once we have the training set and validation set, we can build up the training dataset and validation dataset. 

```python
# build up the training dataset and validation dataset
train_loader = tu_data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = tu_data.DataLoader(val_set, batch_size=64, shuffle=False)
```

The `batch_size` is the number of samples in each batch. The `shuffle` is a boolean value. If `shuffle=True`, the data will be shuffled before each epoch. The idea of having a `batch_size` links to the idea of `mini-batch`. The `mini-batch` is a subset of the training set. The `mini-batch` is used to train the model. The `mini-batch` is randomly sampled from the training set. The `mini-batch` is used to approximate the gradient of the loss function with stochastic gradient descent.

One should be aware that the `batch_size` will affect the training process and the performance of the model. The `batch_size` is a hyperparameter. The `batch_size` is a hyperparameter that we need to tune.

To learn more about the `data loader`, you can check the [documentation](https://pytorch.org/docs/stable/data.html){:target="_blank"}.


After loading the dataset, we need to set up loss function and optimizer. The loss function is used to measure the performance of the model. The optimizer is used to update the parameters of the model.

Here is the full code. 

```python
# %% 
# import packages that are not related to torch
import os
import math
import time
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


#region ###### ---------  environment setup --------- ######

# set up the data path
DATA_PATH = "../data"
SAVE_PATH = "../pretrained"

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#endregion


#region ######  -------- data preprocessing -------- #######
def _get_data():
    """
    download the dataset from FashionMNIST and transfom it to tensor
    """
    # set up the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # download the dataset
    train_dataset = FashionMNIST(root='./data', train=True,
                                    download=True, transform=transform)
    test_dataset = FashionMNIST(root='./data', train=False,
                                    download=True, transform=transform)
    
    return train_dataset, test_dataset


def _visualize_data(train_dataset, test_dataset):
    """
    visualize the dataset by randomly sampling
    9 images from the dataset
    """
    # set up the figure
    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3
    
    for i in range(1, col*row+1):
        sample_idx = np.random.randint(0, len(train_dataset))
        img, label = train_dataset[sample_idx]
        fig.add_subplot(row, col, i)
        plt.title(train_dataset.classes[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
 #endregion       


#region ######  -------- build up neural network -------- #######
class FashionClassifier(nn.Module):

    def __init__(self):
        # initialize the parent class
        # super() allows us to access methods from a parent class
        super(FashionClassifier, self).__init__()
        # define the neural network
        # based on architecture of LeNet-5
        self.net = nn.Sequential(
            # convolutional layers
            # input channel 1 (grayscale), output channel 6
            # kernel size 5 and stride is default 1
            # kernel is just a fancy name for filter
            # here we are using 5 x 5 filter
            # we are using padding to keep the output size the same
            # the output size is (28 - 5 + 2 * 2) / 1 + 1 = 28
            nn.Conv2d(in_channels=1, out_channels=6,
                                     kernel_size=5,
                                     padding=2),
            # activation function sigmoid
            nn.Sigmoid(),
            # average pooling layer
            # the output size is (28 - 2) / 2 + 1 = 14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # the output size is (14 - 5) / 1 + 1 = 10
            nn.Conv2d(in_channels=6, out_channels=16,
                                     kernel_size=5),
            nn.Sigmoid(),
            # the output size is (10 - 2) / 2 + 1 = 5
            nn.AvgPool2d(kernel_size=2, stride=2),
            # dense layers or fully connected layers
            # Flatten is recommended to use instead of view
            nn.Flatten(),
            # lienar layer
            # intput size is 16 * 5 * 5 = 400
            # why 400? because the output size of the last layer is 5 x 5
            # with 16 channels
            # using _layer_summary() to check the output size
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            # output layer, 10 classes
            nn.Linear(in_features=84, out_features=10)
        )


    def forward(self, x):
        return self.net(x)
    
        
    def _layer_summary(self, input_size):
        """
        print the summary of the model
        Input:
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
 #endregion     
           
    
#region ######  -------- function to train the model  -------- #######
def train_the_model(network_model, training_dataset, val_dataset,
                                    num_epochs=13,
                                    patience=7):
    """
    Train the neural network model, we stop if the validation loss
    does not improve for a certain number of epochs (patience=7)
    Inputs:
        network_model: the neural network model
        training_dataset: the training dataset
        val_dataset: the validation dataset
        num_epochs: the number of epochs
        patience: the number of epochs to wait before early stopping
    Output:
        the trained model
    """
    
    # initialize the model
    model = network_model()
    # push the model to the device
    model.to(device)
    
    # hyperparameters setting
    learning_rate = 0.001
    batch_size = 64
    
    # create the loader
    training_loader = tu_data.DataLoader(training_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    validation_loader = tu_data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    
    # define the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # define the optimizer
    # we are using stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9)
    
    # print out the model summary
    print(model)
    
    # loss tracker
    loss_scores = []
    
    # validation score tracker
    val_scores = []
    best_val_score = -1
    
    # begin training
    for epoch in tqdm(range(num_epochs)):
        # set the model to training mode
        model.train()
        correct_preds, total_preds = 0, 0
        for imgs, labels in training_loader:
            # push the data to the device
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # forward pass
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            
            # backward pass
            # zero the gradient
            optimizer.zero_grad()
            # calculate the gradient
            loss.backward()
            # update the weights
            optimizer.step()
            
            # calculate the accuracy
            correct_preds += preds.argmax(dim=1).eq(labels).sum().item()
            total_preds += len(labels)
        
        # append the loss score
        loss_scores.append(loss.item())
        
        # calculate the training accuracy
        train_acc = correct_preds / total_preds
        # calculate the validation accuracy
        val_acc = test_the_model(model, validation_loader)
        val_scores.append(val_acc)
        
        # print out the training and validation accuracy
        print(f"### ----- Epoch {epoch+1:2d} Training accuracy: {train_acc*100.0:03.2f}")
        print(f"                    Validation accuracy: {val_acc*100.0:03.2f}")
        
        if val_acc > val_scores[best_val_score] or best_val_score == -1:
            best_val_score = epoch
        else:
            # one could save the model here
            torch.save(model.state_dict(), SAVE_PATH + "/best_model.pt")
            print(f"We have not improved for {patience} epochs, stopping...")
            break 
        
    # plot the loss scores and validation scores
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].plot([i for i in range(1, len(loss_scores)+1)], loss_scores)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot([i for i in range(1, len(val_scores)+1)], val_scores)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    fig.suptitle("Loss and Validation Accuracy of LeNet-5 for Fashion MNIST")
    fig.subplots_adjust(wspace=0.45)
    fig.show()


def test_the_model(model, val_data_loader):
    """
    Test the model on the validation dataset
    Input:
        model: the trained model
        val_data_loader: the validation data loader
    """
    # set the model to evaluation mode
    model.eval()
    
    correct_preds, total_preds = 0, 0
    for imgs, labels in val_data_loader:
        # push the data to the device
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # no need to calculate the gradient
        with torch.no_grad():
            preds = model(imgs)
            # get the index of the max log-probability
            # output is [batch_size, 10]
            preds = preds.argmax(dim=1, keepdim=True)
            # item() is used to get the value of a tensor
            # move the tensor to the cpu
            correct_preds += preds.eq(labels.view_as(preds)).sum().item()
            total_preds += len(imgs)
    
    test_acc = correct_preds / total_preds
    
    return test_acc
    
 #endregion
 
 
    
def main():
    # download the dataset
    train_dataset, test_dataset = _get_data()
    # _visualize_data(train_dataset, test_dataset)
    
    # split the dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = tu_data.random_split(train_dataset,
                                                      [train_size, val_size])
    train_the_model(FashionClassifier, train_dataset, val_dataset)
    

if __name__ == "__main__":
    print(os.getcwd())
    print("Using torch", torch.__version__)
    print("Using device", device)

    main()
```

Although the code is a bit long, it is much easier to understand than the code written with `numpy`. One can see that the code is much more concise and readable. The code is also much more efficient, as we are using the GPU to train the model. The code is also much more flexible, as we can easily change the model architecture, the optimizer, the loss function, and the hyperparameters. The code is also much more modular, as we can easily reuse the code for other projects.

It takes less than one minute to train the model with one GPU. The model achieves a validation accuracy of 72.38% after 7 epochs.