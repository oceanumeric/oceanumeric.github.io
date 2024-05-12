---
title: GPT in Review - Assessing Performance, Speed, and Cost
subtitle: A special review of the GPT models by OpenAI for certain tasks
layout: blog_default
date: 2024-05-12
keywords: gpt, gpt-3, gpt-4, gpt-5, openai, nlp, natural language processing, transformer, bert, elmo, albert, t5, turing, turing test, performance, speed, cost, review, comparison, benchmark, evaluation, assessment, analysis, summary, conclusion
published: true
tags: chatgpt gpt-3 gpt-4 openai prompt-engineering performance speed cost review evaluation 
---

Ever since the release of GPT-3 by OpenAI, the world has been buzzing with excitement about the capabilities of large language models. I have used API from OpenAI to solve various tasks and developed some applications, such as [a cv job matching tool](https://www.beprepared.studio/){:target="_blank"} and [a firm search and profile generator tool](https://www.hypergi.com/firm-search){:target="_blank"} like perplexity search. 

I have heard many people told me that it is hard to scale the application with GPT models due to many reasons, such as the cost, the speed, and the performance. In this blog post, I will review the GPT models by OpenAI for certain tasks:

- [Summarization based on Goolge search results](#summarization-based-on-goolge-search-results)
- [Ontology generation for a specific domain](#ontology-generation-for-a-specific-domain)
- [Name Entity Recognition for a specific domain](#name-entity-recognition-for-a-specific-domain)

## Summarization based on Goolge search results

During the development of the firm search and profile generator tool, I have used GPT-3.5-turbo-0125 to summarize the search results from Google. The summarization task is to generate a short description of the search results. The input to the model is the search results from Google, and the output is the summary of the search results.

Figure 1 gives the plot of the query time of GPT models for different tokens. The query time increases with the number of tokens, and the heterogeneity of the query time is also observed across different tokens.


<div class='figure'>
    <img src="/blog/images/gpt-querytime-tokens.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Query time of GPT-3.5-turbo-0125 for different tokens.
    </div>
</div>

I am not a premium user of OpenAI, so I do not know whether the query time is the same for all users. Maybe for those who pay more or have a higher usage, the query time is faster. As you can see from the figure 1, the average query time is around 2-3 seconds for any task with more than 300 tokens. One cannot argue that this kind of query time is acceptable for many applications.

<div class='figure'>
    <img src="/blog/images/gpt-querycost-tokens.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Query cost of GPT-3.5-turbo-0125 for different tokens.
    </div>
</div>

The good news is that the cost of the query is not that high. As you can see from the figure 2, the cost of the query is around 0.001 USD for any task with more than 300 tokens. It is actually very cheap to use GPT models for many applications.

There is another thing that I want to mention. __The function calling of the GPT models is not stable__. Sometimes, you got repeated results, and sometimes, you got different formats of results. I do not know why this happens, but it is something that you need to be aware of when you use GPT models. For instance, I set up the format of the results to be a nested json object, but sometimes, GPT models could not return the results in the format that I set up.


I embedded my code including prompt and function calling in the following gist:


<script src="https://gist.github.com/oceanumeric/fd777cbcfa31bbb4fb211ce02a4c3818.js"></script>

## Ontology generation for a specific domain

This is the task that I personally found it very interesting to explore with GPT models. The ontology generation task is to generate a list of concepts and their relationships for a specific domain. The input to the model is the domain-specific text, and the output is the list of concepts and their relationships. 

Overall, the GPT models amazed me with their performance on the ontology generation task, especially GPT-4 models. However, since this task is relatively hard, the query time is way longer than the summarization task. The average query time is 5-6 seconds for any task with more than 300 tokens, which is twice as long as the summarization task. Sometimes, the query time could be longer than 10 seconds, which is not acceptable for many applications. The following figure shows the plot of the query time of GPT models for different tokens comparing to the summarization task.

<div class='figure'>
    <img src="/blog/images/gpt-querytime-tasks.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> Query time of GPT-4-turbo-0125 for different tokens.
    </div>
</div>

It is interesting to notice that 'GPT modesl' behaves like a human being, which means that the _'thinking time is longer for harder tasks'_. For the harder tasks, we observe more heterogeneity in the query time across different tokens. The query time is more stable for the easier tasks, such as summarization.


## Name Entity Recognition for a specific domain

The last task that I want to review is the name entity recognition task. I did many experiments with GPT models on this task because one of my research projects is related to this task. The name entity recognition task is to recognize the names of entities in a specific domain. The input to the model is the news title, and the expected output is firm names, country names, and some properties of the entities, such as whether the firm is a public firm or a private firm. The following table gives some examples of the input and output of the name entity recognition task.


| News Title                                                                                          | Company Name                                     | Location | Is Company | 
|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------|----------|------------| 
| Deputy Director Hu Wei of the Institute of Hydrobiology visited Maotai Group for investigation and exchange | Maotai Group                                     | China    | true       | 
| Institute leaders visited Insect Technology Development Company                                               | Insect Technology Development Company             | China    | true       | 
| Vice President Guo Ying of Sugon Group and his team visited Ganjiang Innovation Institute                  | Sugon Group                                      | China    | true       | 
| Delegation from GlaxoSmithKline visited Institute of Neuroscience                                              | GlaxoSmithKline                                  | Unknown  | false      | 
| Hu Junpeng and his team from Angel Yeast Co., Ltd. visited Yarlong Academy of Sciences and exchanged views | Angel Yeast Co., Ltd.                            | China    | false      | 

The performance of GPT models on the name entity recognition task is truly amazing. The average F1 score is around 0.9 for any task with more than 300 tokens. For my research project, the acuracy of the name entity recognition task is around 0.92, which is very high. For some news titles in Chinese, the accuracy could be lower, which is around 0.89. It is still acceptable for many applications. 

If you have tried other NLP libraries, such as spaCy, or other NER models, such as BERT, you will find that the performance of GPT models is much better than those models. 

In conclusion, the GPT models by OpenAI are very powerful for many NLP tasks. The performance of the models is very high, and the cost of the models is very low. However, the speed of the models is not that fast, especially for harder tasks. The query time is around 2-3 seconds for the summarization task, 5-6 seconds for the ontology generation task, and 1-2 seconds for the name entity recognition task. The query time is more stable for the easier tasks, such as summarization, and more heterogeneity is observed in the query time across different tokens for the harder tasks.

I think the trade-off between the performance, the speed, and the cost will shapre the competition of the GPT models in the future. As we all know, there is no free lunch in the world. How to balance the trade-off is the key to the success of the GPT models.