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
firms in information and communication industry from Germany. I have 
another dataset called `HAN_NAMES`, which includes harmonized names of
patent applicants. Now, I would like to link those two datasets by
matching company names. 


## Peculiarities of company names (strings)

{% katexmm %}

According to Tolstoy, "Happy families are all alike; every unhappy family is unhappy in its own way." This rule can be applied to data. Cleaned datasets
are all alike; every uncleaned dataset is messy in its own way. 

For my case, the dimension of `orbis_de` is $249 \times 19$ and the dimension
of `HAN_NAMES` is $4,191,007 \times 3$. This means I am matching a small
dataset with a relative large one. For this kind of case, it's better to 
do query in the large dataset `HAN_NAMES` instead of using cosine similarity
(see my other post). 


<div class="table-caption">
<span class="table-caption-label">Table 1.</span> Selected names from orbis_de 
dataset.
</div>


|name_international                               |
|:--------------------------------------------:|
|Uniper Energy Storage GmbH                   |
|Worlee-Chemie GmbH                           |
|DEKRA Automobil GmbH                         |
|secunet Security Networks Aktiengesellschaft |
|Zoller & Froehlich GmbH                      |


When I tried to use `%like%` operator in `R` and noticed that names like
_Worlee-Chemie GmbH_ could not be matched but it could be matched after
deleting hyphen symbol: _Worlee Chemie GmbH_. For the last name in 
the Table 1, I got five matches from `HAN_NAMES` dataset shown in the Table
2, which is not as what I expected. 


<div class="table-caption">
<span class="table-caption-label">Table 2.</span> Matched names for Zoller & Froehlich GmbH from HAN_NAMES dataset. 
</div>

|HAN_ID  |Clean_name                                     |Person_ctry_code |
|:-------|:----------------------------------------------|:----------------:|
|3286919 |ZOLLER & FROEHLICH GMBH                        |DE               |
|3425297 |Z F ZOLLER & FROEHLICH GMBH 88239 WANGEN DE    |DE               |
|3519731 |ZOLLER & FROEHLICH GMBH 88239 WANGEN DE        |DE               |
|3738711 |ZOLLER & FROEHLICH GMBH & CO KG 7988 WANGEN DE |DE               |
|3738766 |ZZOLLER & FROEHLICH GMBH                       |DE               |

Here is another example to show peculiarities of matching company names.
For the company _Infineon Technologies Dresden GmbH & Co. KG_, I could 
not any items from `HAN_NAMES` with the original string. Since
_Infineon Technologies Dresden GmbH & Co. KG_ is a very big company, 
I googled it and found _Infineon Tech Dresden GmbH & Co. KG_ is used 
sometimes, and then the exact match was found with string _Infineon Tech Dresden GmbH_.  

{% endkatexmm %}



## Design an algorithm 



Before implementing our own algorithms to match names, I tried a package
called `fedmatch`. The exact math gives 164 rows, which means the 
matching rate is around 65.85%.  However, since I know firms from
`orbis_de` are very large and well-knowns firms and most of them must
hold some patents. Therefore, 90% of those names should be included
in `HAN_NAMES` dataset. 

Here is the algorithm to improve the matching accuracy:

- using `fedmatch` to get those exact matched
- using probability model (Hypergeometric Distribution) to make
inference the probability of _false negative_ in `matched_results$orbis_de_nomatch`. 

{% katexmm %}

Let's model a random variable $X$ has a hypergeometric distribution if $X=x$
is the number of successes in a sample size of size k (without replacement) when the population contains M successes and N non-successes. 

$$
f(x |m, n, k) = \frac{\binom{m}{x} \binom{n}{k-x}}{\binom{m+n}{k}}
$$

In our case

- $k = 5, x=1$
- $M = 10, N=70, M+N=82$

This means we assume there were 10 names are false negative and if we 
draw 5 names out of $82$ names, we want to calculate the probability of having one false
negative in our sample. 

```r
1-phyper(1, m=10, n=70, k=5)  # 0.1151
```
I did run four times of this kind of test, and three of those tests 
have false negative values, this means the probability is around $0.75$, 
which is higher than $0.1151$. 


<div class='figure'>
    <img src="/images/blog/match_hypergeom_prob.png"
         alt="Probability estimation of hypergeometric distribution"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Plot of hypergeometric
        distribution probability for different values of m. 
    </div>
</div>




{% endkatexmm %}
