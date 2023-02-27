---
title: Firm Ownership and The Complexity of Our Economy
subtitle: Multinational corporations have huge influence on our daily life with their global presence; in this post, I will use a dataset from Orbis to trace their footprints. 
layout: blog_default
date: 2023-02-26
keywords: business analytic, CORDIS, Orbis, matching, WRDS, database, string-matching, open-data
published: true
tags: data-science dataset python r business-analytics string-matching 
---


EU Joint Research Center (JRC) published a report called EU Industrial R&D Investment Scoreboard, which analyzes the world top 2500 R&D investors 
and EU top 1000 R&D investors {% cite grassano20212021 %}. In this [report](../blog/pdfs/eu_scoreboard_2021.pdf){:target="_blank"}
I learned that the world top 2500 R&D investors with
headquarters in 39 countries, have more than __800k__ subsidiaries all over the world. 

I was curious how and where they got the number as it shows the strong
presence of multinational corporations (MNCs) globally. In this post, I will
find out who those MNCs and subsidiaries are and where they locate. I will use R to do business analytics and code will be shared here for anyone who
wants to use it as a case study of learning R (or `data.table` package). 


## Top corporate R&D investors 

Table 1 gives the world top 10 R&D investors, which shows IT is the dominant
sector. The dataset from EU JRC also include profit, firm size (number of 
employees) and other firms' characteristics. 


<div class="table-caption">
<span class="table-caption-label">Table 1.</span> World top 10 R&D investors with their R&D intensities and market cap. 
</div>

|Rank |Company                     |   Country   |     Industry    |R&D 2021 (€million) |R&D intensity (%) |Market cap (€million) |
|:----------|:---------------------------|:-----------:|:---------------------------------:|-------------------:|-----------------:|---------------------:|
|1          |ALPHABET                    |     US      |   Software & Computer Services    |27866.85            |12.25             |769312.73             |
|2          |META                        |     US      |   Software & Computer Services    |21768.49            |20.91             |798490.60             |
|3          |MICROSOFT                   |     US      |   Software & Computer Services    |21642.23            |12.36             |2002997.33            |
|4          |HUAWEI INVESTMENT & HOLDING |    China    |  Technology Hardware & Equipment  |19533.85            |16.04             |NA                    |
|5          |APPLE                       |     US      |  Technology Hardware & Equipment  |19348.40            |5.99              |2215940.70            |
|6          |SAMSUNG ELECTRONICS         | South Korea | Electronic & Electrical Equipment |16812.79            |8.08              |340700.57             |
|7          |VOLKSWAGEN                  |   Germany   |        Automobiles & Parts        |15583.00            |6.23              |83746.49              |
|8          |INTEL                       |     US      |  Technology Hardware & Equipment  |13411.62            |19.22             |193644.15             |
|9          |ROCHE                       | Switzerland |  Pharmaceuticals & Biotechnology  |13260.79            |21.83             |249938.43             |
|10         |JOHNSON & JOHNSON           |     US      |  Pharmaceuticals & Biotechnology  |12991.34            |15.69             |402402.89             |


We will now visualize the distribution of firms in terms of industry. 


<div class='figure'>
    <img src="/blog/images/iri_dist_employees.png"
         alt="A demo figure"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Histogram of number of employees and its log transformation; it is important to notice 
        that the left panel figure is right skewed.
    </div>
</div>

One of the stylized facts about firm level dataset is that many univariate 
random variables have the right skewed distribution and the 'bell-shaped'
curve was only visible after doing logarithm transformation. 

<div class='figure'>
    <img src="/blog/images/iri_corr_matrix.png"
         alt="A demo figure"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Correlation matrix 
        shows that there is a strong correlation between  R&D and net sales.
    </div>
</div>

Figure 2 shows the strong correlation exists in a group of variables:
number of employees, R&D expenditure, revenue (net sales), capital expenditure, and profit. The story can be simplified as a narratives: the high R&D spending gives the market power of a firm, which brings higher
revenue; then the firm could invest more on R&D,  upgrade their assets and 
of course hire more talents. 


<div class="table-caption">
<span class="table-caption-label">Table 2.</span> Descriptive statistics grouped by industry, the unit for average revenue per employee is €million. 
</div>


|             industry              |count | average_employee | average_revenue_per_employee |
|:---------------------------------|:-----:|:----------------:|:----------------------------:|
|  Pharmaceuticals & Biotechnology  |478   |     6382.45      |             0.39             |
|   Software & Computer Services    |336   |     12867.55     |             0.31             |
| Electronic & Electrical Equipment |249   |     26253.49     |             0.27             |
|  Technology Hardware & Equipment  |207   |     23051.07     |             1.51             |
|      Industrial Engineering       |167   |     20602.13     |             0.30             |
|        Automobiles & Parts        |148   |     52578.26     |             0.28             |
|             Chemicals             |115   |     16321.54     |             0.55             |
| Health Care Equipment & Services  |89    |     19387.66     |             0.28             |
|     Construction & Materials      |65    |     51241.43     |             1.14             |
|        General Industrials        |64    |     35944.75     |             0.29             |

From Table 2, we can learn that firms from knowledge-intensive industries
hire lots of people and their revenue margin in terms of the size is 
pretty good. 


## Top 1000 EU R&D investors 

In this section, I will search for subsidiaries for top 1000 EU R&D investors. Figure 3 shows that Germany is the leader of innovation in the EU. I will search for firms' subsidiaries based on the rank in the 
Figure 3. 

<div class='figure'>
    <img src="/blog/images/iri_eu_country.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> Top EU R&D investors 
        distribution across countries, which shows Germany is the leading
        country of innovation in the EU. 
    </div>
</div>

Many papers have already tried to find out the subsidiaries of MNCs 
{% cite  ribeiro2010oecd kalemli2015construct clo2020firm grassano20212021 %}. 


<div class='figure'>
    <img src="/blog/images/graph_hist_degree.png"
         alt="A demo figure"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Histogram of degree of
        graph constructed based on the ownership network, which shows 
        the network is very sparse but the 'super hubs' do exist in the 
        ownership network. 
    </div>
</div>
