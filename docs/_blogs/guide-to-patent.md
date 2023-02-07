---
title: Yet Another Guide to Working with EPO Patent Data
subtitle: It took me a while to get into patent analytics, which is a 'wild' world. I hope my guide for patent analytics could help you to navigate through. 
layout: blog_default
date: 2023-02-06
keywords: patent analytic, EPO, PATSTAT, USPTO
published: true
tags: patent patstat api
---

The first question came to my mind was "_where should I start?_" when I
wanted to do patent analytics for my research project. Like everyone, I
ended up with Google or ChatGPT (if you got a chance to ask there). I found 
out that the EPO Worldwide Patent Statistical Database ([PATSTAT](https://www.epo.org/searching-for-patents/business/patstat.html){:target="_blank"})
is one of the most widely used patent databases for researchers and business
analytics. Right now users could get access to PATSTAT online free for one month.
Since most people need to use this database longer than one month,
this guide will cover how you could access the updated PATSTAT database free and
use it whenever you want. 

- [Knowing the data](#knowing-the-data)
- [Getting the dataset](#getting-the-dataset)


## Knowing the data 

Paper by {% cite kang2016patstat %} presented the examples of frequently used patent
statistics in the following table.  

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

- Advantage:
    - Databases based on and linkable to PATSTAT have been produced by other institutions
    - In comparing the characteristics of PATSTAT with the NBER Patent Database of the US Patent data, PATSTAT provides information that are not available in the other two databases such as priority and continuing applications
- Disadvantage:
    - There is a large proportion of missing information in inventors' geographic data
    - PATSTAT has a European-centered bias

There are three most critical issues that users must recognize when planning to use PATSTAT  {% cite kang2016patstat %}:

1. Users need to harmonize names of applicants and inventors; for example, Toyota Motor Corporation appears as “Toyota Motor Corporation,” “Toyota Motor Co,” “Toyota Jidosha Kabushiki Kaisha” (“Jidosha” and “Kabushiki Kaisha” mean a car and a corporation in Japanese, respectively), “Toyota Jidosha K. K.,” and so on.
2. Users need to be aware of the case that one entity can appear with different names in patent documents.
3. Users need to understand how EPO records and tracks its documents, especially when those documents are from different patent offices, such as USPTO. 

With those background information, I believe you are well prepared to start working
with some datasets now. 

## Getting the dataset 

If you want to do some quick search, please try [Espacenet](https://worldwide.espacenet.com/){:target="_blank"},
which uses the same database with PATSTAT. However, as we have mentioned, there
are name issues in this database. Moreover, you cannot do multiple searches
by using `SQL`, `R` or `Python` with Espacenet. PATSTAT does support SQL queries
{% cite  de2014introduction %}, but it is very time consuming to extract information
you want. So, where and how could get the dataset then? 

The starting point is [OECD Harmonized Applicant Names (HAN) Database](https://www.oecd.org/sti/inno/intellectual-property-statistics-and-analysis.htm){:target="_blank"}, which is free for non-commercial use after filling in 
a [short online form](https://forms.office.com/pages/responsepage.aspx?id=1MdBrGEfDUaw9PySWitHHKuxmuqpz_9KusL7-G1D6wFUOEU0OVBYVk5QTzROVlBTSUtBUUREWVhHTiQlQCN0PWcu){:target="_blank"}. 

The OECD HAN database provides a grouping of patent applicant’s names which has been elaborated
with business register data. The names of patent applicants were originally extracted from European Patent
Office’s (EPO) Worldwide Statistical Patent Database (PATSTAT, Spring 2022). The database also includes
the list of patent documents filed to the EPO, the US Patent and Trademarks Office (USPTO) or through the
Patent Co-operation Treaty (PCT). 

Since applicants' names are not consistent (such as Volkswagen Group vs. Volkswagen Shanghai),
OECD patent research group cleaned/harmonized names by matching against company
names from business register data - ORBIS database from Bureau van Dijk (see OECD's [methodology report](/pdfs/OECD HAN Database - August 2022.pdf){:target="_blank"}). 

We have two databases storing different names: PATSTAT and ORBIS. It turned
out that there are _more_ firms in PATSTAT. Therefore, some firms cannot be
matched against company names from ORBIS database. The OECD HAN database, August 2022 (the most updated version I downloaded) has two unique identifiers:

- HAN_id, the number of HAN_id is 4,191,007, which is linked to names that could be matched with ORBIS database
- HARM_id, the number of HARM_id is 4,576,705 (less than that of HAN_id), which is linked to names grouped based on similarity

For instance, if both Volkswagen Group and Volkswagen Shanghai can be matched with
ORBIS database, two HAN_ids will be assigned to them. For names that are similar
with each other, such as "Volksgen (missing 'w') Shanghai", it will be grouped
as Volkswagen Shanghai and only one unique HAN_id will be assigned. 

For HARM_id, unique identifiers are the super set, which not only include
firms that are matched with ORBIS database but also those that cannot
be matched. Figure 1 gives the illustration. 

<div class='figure'>
    <img src="/images/blog/oecd-han-illustrate1.png"
         alt="OECD HAN database illustration"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Relationship
        of HAN_id and HARM_id. 
    </div>
</div>


### OECD HAN database structure 

Figure 2 gives OECD HAN database structure (please open it and check the zoomed version
 with the new tab by right-click).


<div class='figure'>
    <img src="/images/blog/oecd-han-database.png"
         alt="OECD HAN database illustration"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> OECD HAN database structure:
        there are four tables, in which HAN_PERSON is the correspondence
        table which include all cleaned names.
    </div>
</div>


