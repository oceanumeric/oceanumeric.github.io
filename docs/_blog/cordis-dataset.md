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

- Dataset: H2020 - Orbis - OECD Han - EPO office 
- sample size : 93,819 
- sample size of family - 88,834 
- Number of firms: 630 
    - very large company: 241
    - large company: 160
    - medium size: 229 


<div class="table-caption">
<span class="table-caption-label">Table 1.</span> Top 10 companies in Germany
in terms of granted patent holding 
</div>

|Company                     | patent_applications | granted_patents | granted_ratio |
|:---------------------------|:-------------------:|:---------------:|:-------------:|
|bayer ag                    |        11158        |      6831       |     0.61      |
|basf se                     |        9536         |      5576       |     0.58      |
|bayerische motoren werke ag |        5350         |      4076       |     0.76      |
|volkswagen ag               |        4482         |      3187       |     0.71      |
|audi ag                     |        3628         |      2987       |     0.82      |
|infineon tech ag            |        3931         |      2547       |     0.65      |
|zf friedrichshafen ag       |        3711         |      2478       |     0.67      |
|airbus operations gmbh      |        2347         |      1801       |     0.77      |
|continental automotive gmbh |        2831         |      1692       |     0.60      |
|roche diagnostics gmbh      |        2873         |      1554       |     0.54      |



<div class='figure'>
    <img src="/blog/images/patent_ipc_complexity.png"
         alt="A demo figure"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The trend of
        IPC class counts, which shows that patents are tagged 
        with more IPC class. 
    </div>
</div>


<div class='figure'>
    <img src="/blog/images/patent_ai_trend.png"
         alt="A demo figure"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The trend of
        AI patents as the share of total patents. 
    </div>
</div>



<div class="table-caption">
<span class="table-caption-label">Table 2.</span> Top 20 companies in Germany
in terms of AI patents applications (2015-2019)
</div>





|Company                              | AI Patents | AI patents share (%) |
|:-------------------------------------------|:----------:|:----------------:|
|SAP SE                                      |    2589    |      95.29       |
|Infineon Technologies AG                    |    680     |      25.09       |
|Continental Automotive GmbH                 |    489     |      20.55       |
|AUDI Aktiengesellschaft                     |    469     |      21.36       |
|Bayerische Motoren Werke Aktiengesellschaft |    427     |      15.01       |
|Volkswagen Aktiengesellschaft               |    388     |      19.72       |
|Carl Zeiss SMT GmbH                         |    367     |      59.00       |
|Roche Diagnostics GmbH                      |    328     |      53.51       |
|Valeo Schalter und Sensoren GmbH            |    304     |      54.77       |
|Sick AG                                     |    265     |      62.21       |
|Carl Zeiss Microscopy GmbH                  |    228     |      77.55       |
|Carl Zeiss Meditec AG                       |    197     |      70.11       |
|Bundesdruckerei GmbH                        |    190     |      45.56       |
|Aesculap AG                                 |    168     |      77.78       |
|Leica Microsystems CMS GmbH                 |    162     |      94.19       |
|KUKA Deutschland GmbH                       |    136     |      60.71       |
|ZF Friedrichshafen AG                       |    131     |       7.36       |
|Conti Temic microelectronic GmbH            |    123     |      35.96       |
|Giesecke+Devrient Mobile Security GmbH      |    116     |      43.12       |
|OSRAM GmbH                                  |    108     |      17.91       |





<div class='figure'>
    <img src="/blog/images/patent_word_cloud.png"
         alt="A demo figure"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> Word cloud of AI Patents. 
    </div>
</div>




{% endkatexmm %}

