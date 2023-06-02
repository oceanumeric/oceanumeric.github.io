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

The series of posts are based on the video by Andrej Karpathy. What I will do is to add some extra explanations with some intuition behind the concepts and I will also add some code to make it more practical. 

__The post is task-driven, which means that we will always start with simple tasks and we will build on top of them to get to the final goal, which is to train a GPT-2 model from scratch.__ _In each task, I will mark the tool or tool component that we will use to solve it. This way, you can always go back to the post where we introduced the tool or tool component to refresh your memory_.

The task will be annotated with <mark style="background-color:#C5BFDE">this color </mark>, whereas the tool or tool component will be annotated with <mark style="background-color:#CDE7D0">this color </mark>. The reason that I am doing this is because I want to make it easier for you to follow the series of posts and those colors are also used by the orginal paper - [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf){:target="_blank"} - to annotate the different sections of the paper.


## Big picture

If you think about our language, it is a sequence of 'symbols' (words, characters, etc). 
Suppose you think the following setence: "I....". You can complete it in many ways: "I will go to the beach", "I think it will rain", "I am hungry", etc.


What all language models do is to try to predict the next word given the previous words. In our example, the language model will try to predict the next word given the previous word "I". 

In this series of post, intead of predicting the next word, we will predict the next character given the previous characters. For example, given the following characters "I am hu", we want to predict the next character "n". Therefore, we will have two big tasks:

- __Task 1__:  <mark style="background-color:#C5BFDE">Given a sequence of characters, predict the next character using names as the dataset </mark>. 

- __Task 2__:  <mark style="background-color:#C5BFDE">Given a sequence of characters, predict the next character using Shakespeare text as the dataset </mark>.

## N-grams

Before we start with the transformer architecture, we will start with a simple model called n-grams. The <mark style="background-color:#CDE7D0"> n-grams </mark> will be our first tool to solve the task 1. Let's call it <mark style="background-color:#CDE7D0"> Tool-1: n-grams model </mark>. With the n-grams model, we will learn:

- how to prepare the data for the model
- how to transform the data into a format that the model can understand (meaning how to transform the data into matrices)


The n-gram is a probabilistic model that predicts the next character given the previous n-1 characters. Since we are working on characters, the easiest way to convert character into numbers is to use the index of letters in the alphabet. For example, the letter "a" will be 0, the letter "b" will be 1, etc. In English, we have 26 letters, so we will have 26 numbers. However, to mark the beginning and the end of the word, we will add two more symbols `<S>` and `<E>` to the alphabet. Therefore, we will have 28 symbols in total.

Please download the [names dataset](https://github.com/oceanumeric/NLP/blob/main/GPT-2/data/names.txt){:target="_blank"} first. Here is the pyton code:

```python
import torch
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
# n-gram
# read data
words = open('./data/names.txt', 'r').read().splitlines()
words[:10]
# ['emma',
#  'olivia',
#  'ava',
#  'isabella',
#  'sophia',
#  'charlotte',
#  'mia',
#  'amelia',
#  'harper',
#  'evelyn']
# create charater to index mapping
chars = sorted(list(set(''.join(words))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
char_to_idx['<S>'] = len(char_to_idx)
char_to_idx['<E>'] = len(char_to_idx) 
char_to_idx
# {'a': 0,
#  'b': 1,
#  'c': 2,
#  'd': 3,
#  'e': 4,
#  'f': 5,
#  'g': 6,
#  'h': 7,
#  'i': 8,
#  'j': 9,
#  'k': 10,
#  'l': 11,
#  'm': 12,
#  'n': 13,
#  'o': 14,
#  'p': 15,
#  'q': 16,
#  'r': 17,
#  's': 18,
#  't': 19,
#  'u': 20,
#  'v': 21,
#  'w': 22,
#  'x': 23,
#  'y': 24,
#  'z': 25,
#  '<S>': 26,
#  '<E>': 27}

# create 2-grams matrix
N_gram_matrix = torch.zeros((len(char_to_idx), len(char_to_idx)), dtype=torch.int32)
for word in words:
    word = ['<S>'] + list(word) + ['<E>']
    for ch1, ch2 in zip(word, word[1:]):
        id_row = char_to_idx[ch1]
        id_col = char_to_idx[ch2]
        N_gram_matrix[id_row, id_col] += 1

# create dict called idx to char
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
idx_to_char
# {0: 'a',
#  1: 'b',
#  2: 'c',
#  3: 'd',
#  4: 'e',
#  5: 'f',
#  6: 'g',
#  7: 'h',
#  8: 'i',
#  9: 'j',
#  10: 'k',
#  11: 'l',
#  12: 'm',
#  13: 'n',
#  14: 'o',
#  15: 'p',
#  16: 'q',
#  17: 'r',
#  18: 's',
#  19: 't',
#  20: 'u',
#  21: 'v',
#  22: 'w',
#  23: 'x',
#  24: 'y',
#  25: 'z',
#  26: '<S>',
#  27: '<E>'}

# plot the matrix
plt.figure(figsize=(17, 17))
plt.imshow(N_gram_matrix, cmap='Blues')
# add charater labels
for i in range(len(char_to_idx)):
    for j in range(len(char_to_idx)):
        # get charater
        chars = idx_to_char[i] + idx_to_char[j]
        plt.text(j, i, chars, ha='center', va='bottom', color='gray')
        # add number
        plt.text(j, i, N_gram_matrix[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off');
```

<div class='figure'>
    <img src="/images/blog/2-gram-frequency.png"
         alt="2-gram-frequency" class="zoom-img"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The 2-gram frequency matrix. I suggest you zoom in the image to see the numbers(Click on the image to zoom in).
    </div>
</div>

Notice that the last row and the second last column are all zeros. This is because the last character in the word is always `<E>`. Therefore, the last character in the word will never be followed by another character. The same logic applies to the first column and the second row. The first character in the word is always `<S>`, so it will never be preceded by another character.

To solve this problem, we will use `.` to replace `<E>` and `<S>`. The `.` is a special character that represents the beginning and the end of the word. The `.` is not a character in the alphabet, so we will add it to the alphabet. Therefore, we will have 27 symbols in total.

