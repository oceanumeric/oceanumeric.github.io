---
title: Linking Different Databases by Matching Company Names
subtitle: Different datasets very often collected from different sources, e.g. Compustat by S&P, Orbis by Bureau van Dijk, and patent database by different patent authorities. To ensure that the data from different datasets applies to the same company, we need to link those them together either by unique identifiers or by company names. 
layout: blog_default
date: 2023-02-18
keywords: patent analytic, EPO, PATSTAT, USPTO, Orbis, matching, WRDS, database, string-matching
published: true
tags: patent patstat api python r string-matching database 
---


I have a small dataset called `orbis_de` which stores the names of
frims in information and communication industry from Germany. I have 
another dataset called `HAN_NAMES`, which includes harmonized names of
patent applicants. Now, I would like to link those two datasets by
matching company names. 