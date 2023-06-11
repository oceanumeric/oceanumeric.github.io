---
title: Attention is all you need - Part 1
subtitle: The title tells it all and transformers are here to stay. If you want to join the trend of shaping the future of NLP, this is the place to start.
layout: blog_default
date: 2023-06-02
keywords: NLP, transformers, attention, BERT, GPT-2, GPT-3, Pytorch, Huggingface, OpenAI, Deep Learning, Machine Learning, Data Science, Python,
published: true
tags: deep-learning NLP transformers attention  GPT-2 Pytorch Huggingface OpenAI
---


The impact of the transformer architecture in the field of NLP has been huge. It has been the main driver of the recent advances in the field and it is here to stay. In this series of posts, we will go through the main concepts of the transformer architecture and we will see how to use it in practice. Our goal is to train a GPT-2 model from scratch using Pytorch. 

The series of posts are based on the video by Andrej Karpathy. What I will do is to add some extra explanations with some intuition behind the concepts and I will also add some code to make it more practical. To fully understand the transformer architecture, we need to follow the path of the original paper, which means tracing the evolution of the transformer architecture from recurrent neural networks (RNNs) to the final transformer architecture. Here is our roadmap:


1. [Understanding the neurons in the deep learning context](#1-understanding-the-neurons-in-the-deep-learning-context)
2. [Implementing a simple RNN model](#1-implementing-a-simple-rnn-model)
3. Implementing a LSTM and GRU model
4. Implementing a sequence-to-sequence model
5. Implementing a RNN encoder-decoder model
6. Implementing a neural machine translation model
7. Implementing a transformer encoder-decoder model


## 1. Understanding the neurons in the deep learning context

{% katexmm %}

In the deep learning context, we try to represent everything into numbers (float or integer) and when we pass information into our neural network, how those information will be transformed is determined by the weights of the neural network and the activation function. The weights are the parameters of the neural network that are learned during the training process. The activation function is a function that determines the output of the neural network.

<div class='figure'>
    <img src="/blog/images/summary_activation_fn.png"
         alt="activation function"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Three most common activation functions.
    </div>
</div>

The properties of those activation functions make them useful in different kind of situations. For instance, the sigmoid function can be used as a binary classifier or a binary gate as it takes values between 0 and 1. The tanh function is similar to the sigmoid function, but it takes values between -1 and 1. The ReLU function is the most popular activation function and it is used in most of the neural networks. It is a non-linear function and it is very easy to compute. 

When we use different kinds of activation functions, we might call them different terms such as _memory gate_ or _forget gate_. However, they are all the same thing. They are just activation functions. The only difference is that we use them in different contexts. For instance, in the context of RNNs, we call them _memory gate_ or _forget gate_ because they are used to determine how much information we want to keep from the previous time step. In the context of transformers, we call them _attention_ because they are used to determine how much attention we want to pay to each word in the sentence.

Okay, with that being said, let's move on to the next section.


## 1. Implementing a simple RNN model



If you have studied any regression model, and somehow been explored to autoregression models, such as  Autoregressive Moving Average (ARMA), you could pick up the intuition behind RNNs. There is a class of linear time series models with the general form:

$$
y_t = \sum_{i=1}^p \phi_i y_{t-i} + \sum_{i=1}^q \theta_i \epsilon_{t-i} + \epsilon_t
$$

Here, $y_t$ is a time series, $\epsilon_t$ is a white noise, and $p$ and $q$ are the order of the autoregressive and moving average components, respectively. The autoregressive component is the sum of the previous values of the time series, while the moving average component is the sum of the previous values of the white noise.

Recurrent neural networks (RNNs) has the similar form, but instead of using the previous values of the time series, it uses the previous values of the hidden state. The hidden state is a vector that summarizes the information of the previous values of the time series. The hidden state is updated at each time step, and it is used to predict the next value of the time series.


<div class='figure'>
    <img src="/blog/images/rnn_illustration1.png"
         alt="activation function" class="zoom-img"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> RNN illustration (please zoom in to see the details).
    </div>
</div>

The folllowing text was generated by a simple RNN model trained on the Shakespeare dataset. The text length is 94081, the size of parameters is 325181; and it takes around 1 minute and 23 seconds to train the model. 


```bash
Hew, and serelt,
And torthe to chase,
Nnoury beart ie thye reseaving ghicl simy,
And ware sthor,
And py teye,
The ayd love sownst persund,
An thou t yould to eassil got no moresty
Tie il bothery blaise?
On th tome coof mers ais geare.

Lekala worad siof seles?
Tean to nou thy sion or mowith thy bars, all our pone yournow;
A min thy seaviss afo lyou grangute is mabjey,
Whelu toot murfein; wothh be, beande, fremay ingtice,s no fare morigh toun theish;
y aruselo wata astate:
For thay beinose ton the
```

As you can see that it is not very readable. However it is not bad for a model trained within 2 minutes. The reason why it is not very readable is because the model is not very good at capturing the long-term dependencies. The reason why it is not very good at capturing the long-term dependencies is because of the vanishing gradient problem. 

Before I give the full implementation in `Pytorch`, let me list what I have learned from the implementation:

- it is important to align the input and target sequence, especially when we are constructing the batch.
- we are updating the hidden state alogn the sequence length and we are using the last hidden state to predict the next word.

$$
\begin{aligned}
\hat{y}_t &= \text{softmax}(W_{hy}h_t + b_y) \\
h_t &= \text{tanh}(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
\end{aligned}
$$

- Three parameters will be learned during the training process: $W_{hy}$, $W_{hh}$, and $W_{xh}$; and two bias terms: $b_y$ and $b_h$; and the hidden state $h_0$ is initialized as a zero vector. 
- The hidden state will not be learned during the training process. It is just a vector that summarizes the information of the previous values of the time series.
- Instead, the hidden state will be updated from batch to batch.
- It is $W_{hh}$ that is responsible for capturing the long-term dependencies. If $W_{hh}$ is close to zero, then the model will not be able to capture the long-term dependencies.
- Always be careful on the dimension of the input and output of the neural network. 
- Be aware of immutability or mutability of the tensor for different updating process.
- It is __way more efficient to put everything into a class__ and use `nn.Module` to define the neural network. 

Here is the full implementation of the RNN model in `Pytorch`:

```python
# %%
import os
import math
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --------- define our class module --------- ###
class SimpleRNN(nn.Module):

    def __init__(self, data_path, batch_size, seq_length, hidden_size, drop_prob=0.5):
        super().__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        print("::::::::::---------Loading data------::::::::::\n")

        self.text = self.load_data(data_path)
        self.chars = sorted(set(self.text))
        print("Unique characters: ", len(self.chars))
        print(self.text[:100])
        print("::::::::::---------Processing data------::::::::::\n")
        self.encoded_text = self.process_data()

        self.vocab_size = len(self.chars)

        # initialize the hidden state
        self.Wxh = Parameter(torch.Tensor(self.vocab_size, self.hidden_size))
        self.Whh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bh = Parameter(torch.zeros(self.hidden_size))

        nn.init.xavier_uniform_(self.Wxh)
        nn.init.xavier_uniform_(self.Whh)

        # add dropout layer
        self.droupout_layer = nn.Dropout(drop_prob)

        # add the linear layer
        self.lineary_layer = nn.Linear(self.hidden_size, self.vocab_size)

    
    def forward(self, x, h_t):
        """
        x: input, shape = (batch_size, seq_length, vocab_size), here we use one-hot encoding
        h_prev: hidden state from previous cell, shape = (batch_size, hidden_size)
        """
        batch_size, seq_length, vocab_size = x.shape
        hidden_states = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.Wxh + h_t @ self.Whh+ self.bh)
            # h_t.shape = (batch_size, self.hidden_size)
            # h_t.shape = (batch_size, 1, self.hidden_size)
            # do not do h_t = h_t.unsqueeze(1)
            hidden_states.append(h_t.unsqueeze(1))
        
        # concatenate the hidden states
        hidden_states = torch.cat(hidden_states, dim=1)

        # reshape the hidden states
        hidden_states = hidden_states.reshape(batch_size * seq_length, self.hidden_size)

        # apply dropout
        hidden_states = self.droupout_layer(hidden_states)

        # apply the linear layer
        logits = self.lineary_layer(hidden_states)

        # logits.shape = (batch_size * seq_length, vocab_size)
        # h_t was unsqueezed, so we need to squeeze it back

        return logits, h_t.squeeze(1)  # 
    

    def init_hidden(self, batch_size):

        return torch.zeros(batch_size, self.hidden_size)
    

    def train_model(self, epochs, lr=0.001, clip=5):

        # set the model to train mode
        self.train()

        loss_list = []

        # define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):

            # initialize the hidden state
            h_t = self.init_hidden(self.batch_size)
            # push the hidden state to GPU, if available
            h_t = h_t.to(device)

            for x, y in self.create_batches(self.batch_size, self.seq_length):

                # move the data to GPU, if available
                x = x.to(device)
                y = y.to(device)

                # create one-hot encoding
                # do not do x = F.one_hot(x, self.vocab_size).float()
                # it will have a error message: "RuntimeError: one_hot is not implemented for type torch.cuda.LongTensor"
                inputs = F.one_hot(x, self.vocab_size).float()

                # zero out the gradients
                self.zero_grad()

                # get the logits
                logits, h_t = self.forward(inputs, h_t)

                # reshape y to (batch_size * seq_length)
                # we need to do this because the loss function expects 1-D input
                targets = y.reshape(self.batch_size * self.seq_length).long()

                # calculate the loss
                loss = F.cross_entropy(logits, targets)

                # backpropagate
                loss.backward(retain_graph=True)

                # clip the gradients
                nn.utils.clip_grad_norm_(self.parameters(), clip)

                # update the parameters
                optimizer.step()

                # append the loss
                loss_list.append(loss.item())
            
            # print the loss every 10 epochs
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {:.4f}".format(epoch, loss_list[-1]))

        return loss_list


    def load_data(self, data_path):

        with open(data_path, 'r') as f:
            text = f.read()

        return text
    
    def process_data(self):

        # get all the unique characters
        self.vocab_size = len(self.chars)
        print("Vocabulary size: {}".format(self.vocab_size))

        # create a dictionary to map the characters to integers and vice versa
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # print the dictionaries
        print(self.char_to_int)

        # encode the text, torch.long is int64
        self.encoded = torch.tensor(
            [self.char_to_int[ch] for ch in self.text], dtype=torch.long
        )
        # print out the length
        print("Text length: {}".format(self.encoded.size(0)))
        # print the first 100 characters
        print("The text has been encoded and the first 100 characters are:")
        print(self.encoded[:100])

        return self.encoded
    
    
    def create_batches(self, batch_size, seq_length):

        num_batches = len(self.encoded_text) // (batch_size * seq_length)

        # clip the data to get rid of the remainder
        xdata = self.encoded_text[: num_batches * batch_size * seq_length]
        ydata =  torch.roll(xdata, -1)

        # reshape the data
        # this step is very important, because we need to make sure
        # the input and targets are aligned
        # we need to make sure xdata.shape = (batch_size, seq_length*num_batches)
        # because we feed one batch at a time
        xdata = xdata.view(batch_size, -1)
        ydata = ydata.view(batch_size, -1)

        # now we will divide the data into batches
        for i in range(0, xdata.size(1), seq_length):
            xyield = xdata[:, i : i + seq_length]
            yyield = ydata[:, i : i + seq_length]
            yield xyield, yyield


    # function to predict the next character based on character
    def predict(self, char, h=None, top_k=None):
        # assume char is a single character
        # convert the character to integer
        char = torch.tensor([[self.char_to_int[char]]])
        # push the character to the GPU
        char = char.to(device)
        # one-hot encode the character
        inputs = F.one_hot(char, self.vocab_size).float()

        # initialize the hidden state
        if h is None:
            # h.shape = (1, hidden_size)
            # because we only have one character
            h = torch.zeros((1, self.hidden_size))

        # push the hidden state to the GPU
        h = h.to(device)

        # call the model to get the output and hidden state
        with torch.no_grad():
            # get the output and hidden state
            output, h = self(inputs, h)
            # output.shape = (1, vocab_size)

        # get the probabilities
        # dim=1 because we want to get the probabilities for each character
        p = F.softmax(output, dim=1).data

        # if top_k is None, we will use torch.multinomial to sample
        # otherwise, we will use torch.topk to get the top k characters
        if top_k is None:
            # reshape p as (vocab_size)
            p = p.reshape(self.vocab_size)
            # sample with torch.multinomial
            char_next_idx = torch.multinomial(p, num_samples=1)
            # char_next_idx.shape = (1, 1)
            # convert the index to character
            char_next = self.int_to_char.get(char_next_idx.item())
        else:
            # since we have many characters,
            # it is better to use torch.topk to get the top k characters
            p, char_next_idx = p.topk(top_k)
            # char_next_idx.shape = (1, top_k)
            # convert the index to character
            char_next_idx = char_next_idx.squeeze().cpu().numpy()
            # char_next_idx.shape = (top_k)
            # randomly select one character from the top k characters
            p = p.squeeze().cpu().numpy()
            # p.shape = (top_k)
            char_next_idx = np.random.choice(char_next_idx, p=p / p.sum())
            # char_next_idx.shape = (1)
            # convert the index to character
            char_next = self.int_to_char.get(char_next_idx.item())

        return char_next, h

    # function to generate text
    def generate_text(self, char="a", h=None, length=100, top_k=None):
        # intialize the hidden state
        if h is None:
            h = torch.zeros((1, self.hidden_size))
        # push the hidden state to the GPU
        h = h.to(device)

        # initialize the generated text
        gen_text = char

        # predict the next character until we get the desired length
        # we are not feedding the whole sequence to the model
        # but we are feeding the output of the previous character to the model
        # because the the memory was saved in the hidden state
        for i in range(length):
            char, h = self.predict(char, h, top_k)
            gen_text += char

        return gen_text
    

if __name__ == "__main__":
    print("Hello World")
    print(os.getcwd())

    seq_length = 100
    batch_size = 128   # 512
    hidden_size = 512  # or 256
    epochs = 300
    learning_rate = 0.001
    
    rnn_model = SimpleRNN(data_path="data/sonnets.txt", batch_size=batch_size,
                            seq_length=seq_length, hidden_size=hidden_size)
    # print out number of parameters
    print(f"Number of parameters is {sum(p.numel() for p in rnn_model.parameters())}")
    # push to GPU
    rnn_model.to(device)
    # train the model
    loss_list = rnn_model.train_model(epochs=epochs, lr=learning_rate)
    # generate text
    print(rnn_model.generate_text(char="H", length=500, top_k=5))
```



{% endkatexmm %}





