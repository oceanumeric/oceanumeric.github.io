---
title: Yet Another Guide to Working with Patent Data
subtitle: It took me a whilte to get into patent analytics, which is a 'wild' world. I hope my guide for patent analytics could help you to nagivate through. 
layout: blog_default
date: 2023-02-06
keywords: patent analystics, EPO, PATSTAT, USPTO
published: true
tags: patent patstat api 
---

The first question came to my mind was "_where should I start?_" when I
wanted to do patent analytics for my research project. Like everyone, I
ended up with Google or ChatGPT (if you got a chance to ask there). I found 
out that the EPO Worldwide Patent Statistical Database (PATSTAT) 
is one of the most widely used patent databases for researchers and business
analytics. However, PATSTAT is not free and one has to pay for it. Even
you had bought PATSTAT, you still need a guide to working with it. 

Reading this guide will save you days to working with patent data. hello hsdg

sdg 

- [Knowing the data](#knowing-the-data)
- [Getting the dataset](#getting-the-dataset)


## Knowing the data 

Paper by {% cite kang2016patstat %} presented the examples of frequently used patent
statistics explained in the following table.  

| Analysis 	| Patent statistics 	| Purposes 	|
|---	|---	|---	|
| Citation analysis 	| Forward citations 	| Measure technological value 	|
|  	| Backward citations 	| Find knowledge source 	|
| Patent counts analysis 	| Patent counts 	| Observe patent portfolio 	|
|  	| RTA (Revealed Technology Advance) PS (Patent Share) 	| Identify core technological competence 	|
| Technology class analysis 	| Generality 	| Measure endogenous applicability to different technological fields 	|
|  	| Originality 	| Measure knowledge absorption from different technological fields 	|
| Inventor analysis 	| Inventor counts 	| Measure invention quality Measure absorptive capability 	|
|  	| Inventor 	| Identify specific inventors' info such as star engineers Follow mobility of R&D personnel 	|

In terms of discussed advantages and disadvantages of using PATSTAT data, they
made a summary as follows:

- advantage
    - Databases based on and linkable to PATSTAT have been produced by other institutions
    - In comparing the characteristics of PATSTAT with the NBER Patent Database of the US Patent data, PATSTAT provides information that are not available in the other two databases such as priority and continuing applications. 
- disadvantage
    - There is a large proportion of missing information in inventors' geographic data
    - PATSTAT has a European-centered bias

There are three most critical issues that users must recognize when planning to use PATSTAT  {% cite kang2016patstat %}:

1. users need to harmonize names of applicants and inventors; for example, Toyota Motor Corporation appears as “Toyota Motor Corporation,” “Toyota Motor Co,” “Toyota Jidosha Kabushiki Kaisha” (“Jidosha” and “Kabushiki Kaisha” mean a car and a corporation in Japanese, respectively), “Toyota Jidosha K. K.,” and so on.
2. users need to be aware of the case that one entity can appear with different names in patent documents.
3. users need to understand how EPO records and tracks its documents, especially when those documents are from different patent offices, such as USPTO. 

With those background information, I believe you are well prepared to start working
with some datasets now. 

## Getting the dataset 