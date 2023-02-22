---
title: CORDIS - EU Research Projects Datasets
subtitle: The dataset contains all projects funded by the European Union from 1998 to 2027, I organized those datasets in this post for research purpose. 
layout: blog_default
date: 2023-02-21
keywords: business analytic, CORDIS, Orbis, matching, WRDS, database, string-matching, open-data
published: true
tags: data-science dataset python r business-analytics string-matching 
---

The Community Research and Development Information Service (CORDIS) is the European Commission's primary source of results from the projects funded by the EU's framework programs for research and innovation, from FP1 to Horizon Europe.

I have downloaded the following datasets:

- CORDIS - EU research projects under FP5 (1998-2002)
- CORDIS - EU research projects under FP6 (2002-2006) 
- CORDIS - EU research projects under FP7 (2007-2013)
- CORDIS - EU research projects under Horizon 2020 (2014-2020)
- CORDIS - EU research projects under HORIZON EUROPE (2021-2027)

You can download all datasets from [kaggle](https://www.kaggle.com/datasets/oceanumeric/cordis){:target="_blank"}.



## Linked dataset

{% katexmm %}


With firm information from Cordis, I will link it with Orbis datasets
by matching firm names. Then the matched dataset will be linked to
the OECD HAN datasets by matching company names again. 

The code in the following block gives the algorithm of doing this. There are some important facts about the dataset that one should be 
aware of:

- sample size of H2020 for German firms is 7395 at project level
- sample size of H2020 for German firms is 3339 at firm level (meaning one firm could have several projects going on)
- the sample size of exact matching with Orbis dataset decreases to 1525 (27% - large firms, 73% - SME firms)
- the sample size of exact matching with HAN_NAMES dataset decreases
to 853 (283 of them are large firms)

This means that we lost half of observations each time we linked
datasets by matching names for those datasets. 

After linking with patent dataset by setting each patent saved 
as one entry with its linked information, the sample size grows to
$256813$ with 853 unique firms. 

To extract all patent information for $255813$ patents, it takes several 
days. 


## Data analysis

The dimension of our sample dataset is $53138 \times 46$, which includes
firms' research projects and their patents applied for in EPO's office. However, many patents belong
to the same family, the sample size drops to $50543 \times 46$ if we 
filter patents by unique family IDs. This is very important as we 
do not want to over count patents. 

We will analyze those data by showing what they did in the past through patent information and 
what they will do in the future through their research projects. 

Right now, we only have $177$ firms in our sample. Table 1 lists the
top firms of having the highest patent granted ratio. 





|Company                         | patent_applications | granted_patents | granted_ratio |
|:-------------------------------|:-------------------:|:---------------:|:-------------:|
|bayer ag                        |        11158        |      6831       |     0.61      |
|basf se                         |        9536         |      5576       |     0.58      |
|bayerische motoren werke ag     |        5350         |      4076       |     0.76      |
|audi ag                         |        3628         |      2987       |     0.82      |
|airbus operations gmbh          |        2347         |      1801       |     0.77      |
|continental automotive gmbh     |        2831         |      1692       |     0.60      |
|continental teves ag and co ohg |        1914         |      1434       |     0.75      |
|bundesdruckerei gmbh            |         947         |       732       |     0.77      |
|covestro deutschland ag         |        1225         |       637       |     0.52      |
|aesculap ag                     |         772         |       616       |     0.80      |


AI related .

ad 

dg 




|Company                     | patent_applications | granted_patents | granted_ratio |
|:---------------------------|:-------------------:|:---------------:|:-------------:|
|aesculap ag                 |         520         |       413       |     0.79      |
|audi ag                     |         467         |       362       |     0.78      |
|carl zeiss ag               |         540         |       351       |     0.65      |
|bundesdruckerei gmbh        |         420         |       331       |     0.79      |
|bayerische motoren werke ag |         537         |       328       |     0.61      |
|continental automotive gmbh |         508         |       253       |     0.50      |
|carl zeiss smt gmbh         |         367         |       197       |     0.54      |
|carl zeiss microscopy gmbh  |         226         |       172       |     0.76      |
|bayer ag                    |         388         |       167       |     0.43      |
|carl zeiss meditec ag       |         203         |       140       |     0.69      |








{% endkatexmm %}

