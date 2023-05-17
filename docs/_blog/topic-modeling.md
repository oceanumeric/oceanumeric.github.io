---
title: Topic Modeling with Python
subtitle: A quick guide on how to do topic modeling with Python
layout: blog_default
date: 2023-05-17
keywords: company financial report, data science, research policy
published: true
tags: data-science python research-policy
---

Topic modeling is a technique for discovering the abstract "topics" that occur in a collection of documents. It is a frequently used text-mining tool for discovery of hidden semantic structures in a text body.

I mean no body could read _many_ documents like financial reports in a short time. Topic modeling could help us to understand the main topics in a collection of documents. In this post, I will try to show you how to do topic modeling with R and Python.

## The messy documents

One of my students is working on a project about corporate sustainability. She collects around 10-15 firms of sustainability reports. However the dataset is not balanced as some firms have more reports than others, which is quite common in the real world. The report is in txt format and there are some uncoded characters in the text, which makes the text messy.

Python or R? Here is the rule of thumb:

- if the dataset is table format, use R
- if the dataset is not table format, use Python

## Topic modeling with Python 

Since our dataset is not in table format, we will use Python to do topic modeling. We will use `gensim` package to do topic modeling.

```py
# %%
import os 
from pprint import pprint
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel


# change working directory
# os.chdir("./sustainability/")


def _read_txt(txt_path):
    with open(txt_path, "r") as f:
        txt = f.read()
    return txt


def _read_all_txt(txt_dir):
    txt_files = os.listdir(txt_dir)
    txt_files = [f for f in txt_files if f.endswith(".txt")]
    txt_files = [os.path.join(txt_dir, f) for f in txt_files]
    txt_corpus = [_read_txt(f) for f in txt_files]
    return txt_corpus


def _preprocess_txt(txt_corpus):
    
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(txt_corpus)):
        txt_corpus[idx] = txt_corpus[idx].lower()  # Convert to lowercase.
        txt_corpus[idx] = tokenizer.tokenize(txt_corpus[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    txt_corpus = [[token for token in doc if not token.isnumeric()] for doc in txt_corpus]

    # Remove words that are only one character.
    txt_corpus = [[token for token in doc if len(token) > 1] for doc in txt_corpus]

    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    txt_corpus = [[lemmatizer.lemmatize(token) for token in doc] for doc in txt_corpus]

    # add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(txt_corpus, min_count=20)
    for idx in range(len(txt_corpus)):
        for token in bigram[txt_corpus[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                txt_corpus[idx].append(token)
    
    # remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(txt_corpus)

    # Filter out words that occur less than 5 documents, or more than 70% of the documents.
    dictionary.filter_extremes(no_below=5, no_above=0.7)

    # Bag-of-words representation of the documents.

    corpus = [dictionary.doc2bow(doc) for doc in txt_corpus]

    return corpus, dictionary


def train_lda_model(corpus, dictionary):
    # Set training parameters.
    num_topics = 6
    chunksize = 2000
    passes = 20
    iterations = 300
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    return model




if __name__ == "__main__":
    print("Hello World!")
    print(os.getcwd())

    foo = _read_all_txt("./")

    foo_corpus, foo_dict = _preprocess_txt(foo)

    foo_model = train_lda_model(foo_corpus, foo_dict)

    top_topics = foo_model.top_topics(foo_corpus)  #

    pprint(top_topics)
```

Since the number of document is small, the topic extracted is not very meaningful. However, it still can tell us something useful.