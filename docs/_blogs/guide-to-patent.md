---
title: Yet Another Guide to Working with EPO Patent Data
subtitle: It took me a while to get into patent analytics, which is a 'wild' world. I hope my guide for patent analytics could help you to navigate through. 
layout: blog_default
date: 2023-02-06
keywords: patent analytic, EPO, PATSTAT, USPTO
published: true
tags: patent patstat api python r
---


The first question came to my mind was "_where should I start?_" when I
wanted to do patent analytics for my research project. Like everyone, I
ended up with Google or ChatGPT (if you got a chance to ask there). I found 
out that the EPO Worldwide Patent Statistical Database ([PATSTAT](https://www.epo.org/searching-for-patents/business/patstat.html){:target="_blank"})
is one of the most widely used patent databases for researchers and business
analytics. Right now users could get access to PATSTAT online free for one month.
Since most people need to use this database longer than one month,
this guide will cover how you could access the PATSTAT database for __free__ and
use it whenever you want. 

- [Knowing the data](#knowing-the-data)
- [Getting the dataset](#getting-the-dataset)
- [Understanding OECD HAN database structure](#understanding-oecd-han-database-structure)
- [A case study with OECD HAN datasets](#a-case-study-with-oecd-han-datasets)
    - [Read the dataset](#read-the-dataset)
    - [Search for granted patents](#search-for-granted-patents)
    - [Linked open EP data](#linked-open-ep-data)


## Knowing the data 

Paper by {% cite kang2016patstat %} presented the examples of frequently used patent
statistics in the following table.  

<div class="table-caption">
<span class="table-caption-label">Table 1.</span> What we can do with patent information {% cite kang2016patstat %}. 
</div>

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

Since applicants' names could have some level of similairity  but still refers to the different entity (such as Volkswagen Group vs. Volkswagen Shanghai), we need to identify
those entities by matching the official business register database. OECD patent research group cleaned/harmonized names by matching against company
names from business register data - ORBIS database from Bureau van Dijk. 

Sometimes, names have typos but computer could not identify those names as 
the same entity such as "Volksgen (missing 'w') Shanghai". For cases like
this, we need to clean and harmonized names again. 

Please read OECD's [methodology report](/pdfs/OECD HAN Database - August 2022.pdf){:target="_blank"} to know how they clean, harmonize and consolidate applicant
names.  


## Understanding OECD HAN database structure 

Figure 1 gives OECD HAN database structure (open the zoomed version
for more details). It has four tables: `HAN_PERSON`, `HAN_NAMES`,
`HARM_NAMES`, and `HAN_PATENTS`. We will use one patent file as an example to
walk through those tables. Notice that there are around 18 million rows in
`HAN_PATENTS` table, which accounts for around 18% of EPO's total patent document.[^1]
This means the OECD HAN database has a European-centered bias. This is partly
due to the useage of ORBIS database in the matching process,  which has more register information for European firms.
We will discuss how to get and analyze patent data for Chinese firms in the
coming sections of this post. 


[^1]:there are more than 100 million patent documents from leading industrialized and developing countries in EPO's database.



<div class='figure'>
    <img src="/images/blog/oecd-han-database.png"
         alt="OECD HAN database illustration" class="zoom-img"
         style="width:100%"
         />
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> OECD HAN database structure:
        there are four tables, in which HAN_PERSON is the correspondence
        table which include all cleaned names.
    </div>
</div>

Now, we chose the first six records (or entries) from `HAN_PATENTS` table and search
those patents from EPO's [Espacenet](https://worldwide.espacenet.com/?locale=en_EP){:target="_blank"}.
In the `HAN_PATENTS` table, the variable `Appln_id` refers to the patent application
identifier in PATSTAT database. You can use it to retreat the document if you
can access to PATSTAT database. When we search in Espacenet, we use `Patent_number`
instead of `Appln_id`. 

<div class="table-caption">
<span class="table-caption-label">Table 2.</span> First six rows of HAN_PATENTS, which starts at HAN_ID = 4. 
</div>

| HAN_ID <int> | HARM_ID <int> | Appln_id <int> | Publn_auth <chr> | Patent_number <chr> |
|--------------|---------------|----------------|------------------|---------------------|
| 4            | 4             | 311606173      | US               | US8668089           |
| 7            | 7             | 439191607      | US               | US9409947           |
| 7            | 7             | 518367793      | US               | US10836794          |
| 10           | 10            | 365204276      | US               | US8513480           |
| 14           | 14            | 336903179      | WO               | WO2011112122        |
| 14           | 14            | 363622722      | WO               | WO2012064218        |

Let's search for `US8668089` - the first entry with `HAN_ID=4`.
The following search result was returned by Espacenet. 

<div class='figure'>
    <img src="/images/blog/US8668089.png"
         alt="Searching results for US8668089"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Espacenet's searching
        returned result that includes patent name, inventor, applicant, CPC classes,
        and publication information and priority date. 
    </div>
</div>

To understand meaning of those dates, code and numbers, we need to learn some
legal terms in the field of patent analytics. 

- Patent __application__:  The __formal "paperwork"__ filed by an applicant, 
(or by a patent attorney or patent agent on the applicant’s behalf)
seeking to obtain a patent for a specific invention. There are
several parts to a patent application, for example the
description and the claims. In the above example, the patent file called 
_System Crate, in Particular for Transporting Fresh Fish_ is the formal 'paperwork'. 

- Patent __priority__:  The priority date is the first date of filing of a 
patent application. It is essential for determining whether any subsequent application for the
same invention can still be assessed as novel. It also makes it possible to
determine whether the subject-matter of a patent application is prior art
on a particular date. The priority date is, however, not necessarily the same as the filing date.
    - in our case, the priority date is __2006-05-29__.
    - if another firm filed the patent application with the similar idea later than 2006-05-29, the subsequent
    application will be assessed as non-novel. 
    - understanding priority and prior arts can be tricky, please read my conversation with 
    [ChatGPT](https://oceanumeric.github.io/blogs/talk-with-chatGPT-about-patent){:target="_blank"}. 

- Patent __publication__: The first patent publication is often 
the __published patent application__, EPO offices published the patent file 
to the public 18 months after a priority date. USPTO has different rules.
In this example, the publication date is 2010-04-22 (that's why publication
number starts `US2010....(A1)`) The code `A1` in the publication number refers to patent application publication kind.
Sometimes, firms might update the application and they will republish it again
with the same name, then A2 will be used in the publication code. Those kind code
(as explained in the next bullet point) have slight different meanings
in different patent office (but ideas behind those code are same). 

- Patent publication __kind code__: A code which includes 1 or 2 letters and in many cases a
number, used to distinguish the kind of published patent
document. For example, the publication of an application used in EPO for a
patent with or without a search report, and the level of the
publication:
    - A1: European patent application published with European search report
    - A2: European patent application published without European search report (search report not available at publication date)
    - A3: document Separate publication of the European search report
    - A4: document Supplementary search report
    - B1: European patent specification (granted patent)
    - B2: New European patent specification (amended specification after opposition procedure)

- Patent application __claims__: The part of the patent that defines the scope of the legal
protection sought for the invention. 

- Patent __granted__: the patent was granted on 2014-03-11 for our example
_System Crate, in Particular for Transporting Fresh Fish_, with a new 
document number `US8668089 (B2)`. 

- Patent __citation(s)__: A patent document cited. Citations are not only added by the
patent applicant but also by the examiners of the patent
application. Patents citations may be added during the
different steps of the granting process (search report,
examination, third party observations, opposition) and thus
added to the patent data.

- Simple patent __family__: All documents sharing exactly the same set of priorities.

- Patent __family__: All documents sharing directly or indirectly at least one
priority.


One has to be very __careful__ when you are working with patent data because 
you might have different _reference IDs_ pointing to the same patent. Therefore,
it is better to work only on granted patents unless you want to do patent
analysis based on application documents. Having a grip on it by looking
at the Figure 3. 

<div class='figure'>
    <img src="/images/blog/patent-search-illustration.png"
         alt="Searching results illustated for US8668089" 
         class="zoom-img"
         style="width:80%; margin: 0 auto;"
         />
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The complexity of
        a patent file; read the granted patent file <a href="/pdfs/US_8668089_B2.pdf" target="_blank">[pdf]</a>
        to have a big picture of this patent file.
    </div>
</div>

It makes sense for a firm to file a patent in different judicial offices, therefore
you see the patent in Figure 3 with publication number US2010096288(A1) was granted
in different offices and then different document numbers such as `US8668089(B2)`,
`EP2024242(B1)`. This might lead to the issue of overcounting patents if we
do not search cautiously. 

Let's review the first from `HAN_PATENTS`, which we have extracted the
patent document for the first entry with patent number of `US8668089`. Notice that the starting entry in `HAN_PATENTs` table is at `HAN_ID=4` 
instead of 1, this means for companies with `HAN_ID=1` (or 2-3), 
there is no patent information for them.

| HAN_ID <int> | HARM_ID <int> | Appln_id <int> | Publn_auth <chr> | Patent_number <chr> |
|--------------|---------------|----------------|------------------|---------------------|
| __4__           | 4             | 311606173      | US               | __US8668089__          |


Table 3 presents the first seven rows of `HAN_PERSON`. Compared with Table 2,
it starts at `HAN_ID=1`, this means some names picked by algorithms do not have
patent files even though they have unique person IDs. In this table, `Person_id`
refers to applicant's unique identifier from PATSTAT. This table is of little
misleading as many of them are firms' names. So, be aware that `Person_name_clean`
refers to the harmonised applicant name. 

<div class="table-caption">
<span class="table-caption-label">Table 3.</span> First seven rows of HAN_PERSON, which starts at HAN_ID = 1. 
</div>

| HAN_ID | HARM_ID | Person_id |               Person_name_clean | Person_ctry_code | Matched |
|:-------:|:--------:|:----------:|--------------------------------:|:-----------------:|:--------:|
|  <int> |   <int> |     <int> |                           <chr> |            <chr> |   <int> |
|      1 |       1 |  15164069 | & HAMBOURG NIENDORF             |               DE |       0 |
|      2 |       2 |  30816597 | & KK                            |               JP |       0 |
|      3 |       3 |  67383967 |           “ASTRONIT” CLOSE CORP |               RU |       0 |
|      3 |       3 |  47772459 |           “ASTRONIT” CLOSE CORP |               RU |       0 |
|      __4__ |       4 |  __47210990__ |             __“DEUTSCHE SEE” GMBH__ |               __DE__ |       0 |
|      5 |       5 |  47367244 |  “EFIRNOIE” OPEN JOINT STOCK CO |               RU |       0 |
|      5 |       5 |  64495153 |  “EFIRNOIE” OPEN JOINT STOCK CO |               RU |       0 |


For the first four rows,
here are possible reasons why they are included in Table 3 but not in Table 2:

- _HAMBOURG NIENDORF_ is a place, somehow PATSTAT identifies it as applicant; this
might be due to the misspecification of software or human beings; for instance, somehow 
a name of place was filled or saved in the place setted for applicant.
- _&KK_, this misspecification is very likely caused by the algorithm which might treat
any strings after `&` as a name
- _“ASTRONIT” CLOSE CORP_, this one has two different `Person_id` but same `HAN_ID`
because the name is same but applicant identifiers are different. 


__Differences between `HAN_ID` and `HARM_ID`__. In Table 4, we have selected four names with same `HAN_ID` but different `HARM_ID`s. This shows that `HAN_ID` was
grouped based on further harmonized process. Clearly, _“PERMNEFTEMASHREMONT” PUBLIC JOINT STOCK CO_ is very likely to referring the same entity with _PUBLIC JOINT STOCK CO “PERMNEFTEMASHREMONT”_. 

<div class="table-caption">
<span class="table-caption-label">Table 4.</span> The selected rows from `HAN_PERSON`
database. 
</div>

| HAN_ID | HARM_ID | Person_id |                           Person_name_clean | Person_ctry_code | Matched |
|:-------:|:--------:|:----------:|--------------------------------:|:-----------------:|:--------:|
|  <int> |   <int> |     <int> |                                       <chr> |            <chr> |   <int> |
|      6 |       6 |  67613311 |             “EUROSTANDART” LTD LIABILITY CO |               RU |       0 |
|      7 |       7 |  53655360 |                                  “IVIX” LTD |               RU |       0 |
|      8 |       8 |  48761393 | “PERMNEFTEMASHREMONT” PUBLIC JOINT STOCK CO |               RU |       1 |
|      8 |       8 |  64875050 | “PERMNEFTEMASHREMONT” PUBLIC JOINT STOCK CO |               RU |       1 |
|      8 | 2462366 |  48754911 | PUBLIC JOINT STOCK CO “PERMNEFTEMASHREMONT” |               RU |       1 |
|      8 | 2462366 |  64543597 | PUBLIC JOINT STOCK CO “PERMNEFTEMASHREMONT” |               RU |       1 |


In the next section, we will do a case study to link a small dataset with
OECD HAN datasets and try to answer some interesting analytical questions. 

## A case study with OECD HAN datasets 

{% katexmm %}

From now on, we will only work on three datasets: `HAN_NAMES`, `HAN_PERSON` and `HAN_PATENTS`
as those two datasets include all information we need. With those two datasets
we will try to do patent analysis for some german firms from information 
and communication technology industry. Here are components of this case
study: 

- Inputs:
    - dataset 1: a sample of German firms, size $N=246$
    - dataset 2: `HAN_NAMES`, size $N=4,191,007$
    - dataset 3: `HAN_PATENTS`, size $N=18,355,687$
- Goal:
    - match names of those german firms with `HAN_NAMES` and then find their 
    _granted_ patents from `HAN_PATENTS`[^2]
- Actions:
    - read dataset 1 and 2
    - try to find out whether firms from dataset 1 are included in dataset 2 or not 
    - if it was included, then use `HAN_ID` to extract patents for corresponding firms
    from dataset 3
- Outcomes:
    - a patent analytical report


[^2]: `HAN_PATENTS` include both patent application documents (A1 or A2) and granted patents (B1 or B2), therefore we need to figure out how to extract granted patents. 


### Read the dataset

Table 5 gives the top rows of our dataset `german_firms.csv`, which
has firms' native names and international names. We will use the column
of `name_international` as native names might have special german 
characters like ö or ü. 

<div class="table-caption">
<span class="table-caption-label">Table 5.</span> Top rows
of German firms dataset.
</div>


|              name_native              |             name_international             |
|:-------------------------------------:|:-------------------------------------:|
|     Airbus Defence and Space GmbH     |     Airbus Defence and Space GmbH     |
|                EurA AG                |                EurA AG                |
|        TuTech Innovation GmbH         |        TuTech Innovation GmbH         |
| FFT Produktionssysteme GmbH & Co. KG. | FFT Produktionssysteme GmbH & Co. KG. |
|     Diehl Aviation Laupheim GmbH      |     Diehl Aviation Laupheim GmbH      |

Table 6 gives the top rows of `HAN_NAMES`. Our goal is to extract
`HAN_ID` based on the value of `Clean_name`. For instance, we
will try to find `Airbus Defence and Space GmbH` in `HAN_NAMES`. 

<div class="table-caption">
<span class="table-caption-label">Table 6.</span> Top rows
of `HAN_NAMES`.
</div>

| HAN_ID |           Clean_name           | Person_ctry_code |
|:------:|:------------------------------:|:----------------:|
|   1    |      & HAMBOURG NIENDORF       |        DE        |
|   2    |              & KK              |        JP        |
|   3    |     “ASTRONIT” CLOSE CORP      |        RU        |
|   4    |      “DEUTSCHE SEE” GMBH       |        DE        |
|   5    | “EFIRNOIE” OPEN JOINT STOCK CO |        RU        |

Notice that all names in `HAN_PATENTS` are of upper case, we need
to convert our german firms' names into upper cases too. Then 
we can do our query. 

```r
# Code Block 1. query names in han_names 
airbus <- toupper('Airbus Defence and Space GmbH')
han_names %>%
    .[Clean_name %like% airbus]
```

The above code returns nothing as we could not find the _exact_ match.
Therefore, we need to query with less characters. 

```r
# Code Block 2. query names in han_names with less characters   
airbus <- toupper('Airbus Defence')
han_names %>%
    .[Clean_name %like% airbus]
```

When we query with `AIRBUS DEFENCE`, we got more than 10 results shown in Table 7. This shows the challenging part of doing patent analysis[^3]. Have a look at the one with `HAN_ID=62422`, you will realize that why we could not find anything when we use `'AIRBUS DEFENCE AND SPACE GMBH'` to do query because there is no '_AND_' but
an '_ANG_'. Moreover, we have same firm registered in different
countries. 

[^3]: actually this kind of problem is very common in business analytics or any text analysis related to identifying entities' names. 

<div class="table-caption">
<span class="table-caption-label">Table 7.</span> The results of the above query
</div>

|HAN_ID  |                            Clean_name| Person_ctry_code |
|:-------:|:-------------------------------------|:----------------:|
|53993   |                AIRBUS DEFENCE & SPACE|        FR        |
|60513   |           AIRBUS DEFENCE & SPACE GMBH|        DE        |
|60514   |            AIRBUS DEFENCE & SPACE LTD|        GB        |
|61747   |           AIRBUS DEFENCE & SPACE GMBH|        DD        |
|61749   |            AIRBUS DEFENCE & SPACE SAS|        FR        |
|62421   |            AIRBUS DEFENCE & SPACE LTD|        FI        |
|_62422_   |         _AIRBUS DEFENCE ANG SPACE GMBH_|        _DE_        |
|68676   |           AIRBUS DEFENCE & SPACE GMBH|        GB        |
|81263   | AIRBUS DEFENCE & SPACE NETHERLANDS BV|        NL        |
|3637004 |                AIRBUS DEFENCE & SPACE|        DE        |
|4401227 |         AIRBUS DEFENCE ANS SPACE GMBH|        DE        |
|4527012 |       AIRBUS DEFENCE & AIR SPACE GMBH|        DE        |


Since we only want German firms, we will query with the
extra row filter by setting `Person_ctry_code == 'DE'`. This
gives us the following results, which means we will use all 
those five `HAN_IDs` to extract patent information for Airbus Defence and Space GmbH. 

<div class="table-caption">
<span class="table-caption-label">Table 8.</span> The results for querying German firms
</div>

| HAN_ID  |Clean_name                      | Person_ctry_code |
|:-------:|:-------------------------------|:----------------:|
|  60513  |AIRBUS DEFENCE & SPACE GMBH     |        DE        |
|  62422  |AIRBUS DEFENCE ANG SPACE GMBH   |        DE        |
| 3637004 |AIRBUS DEFENCE & SPACE          |        DE        |
| 4401227 |AIRBUS DEFENCE ANS SPACE GMBH   |        DE        |
| 4527012 |AIRBUS DEFENCE & AIR SPACE GMBH |        DE        |

By the way, thought I like `Python` a lot, I have to admit that
the workflow of doing data analysis with packages like `data.table`
or `tidyverse` is much better in `R` community when your dataset is quite organized
and cleaned. The pipe function `%>%` is one of my favorite. 

```r
# Code-Block 3. Filter and extract han_ids for Airbus Defence
airbus <- toupper('Airbus Defence')
han_names %>%
    .[Person_ctry_code == 'DE'] %>%
    .[Clean_name %like% airbus] %>%
    .[,HAN_ID] -> airbus_han_ids
airbus_han_ids
#[1] 60513 62422 3637004 4401227 4527012

# Code-Block 4. Count patents for different offices 
han_patents %>%
    .[HAN_ID %in% airbus_han_ids] %>%
    .[, .N, by=Publn_auth] -> foo
    transform(adorn_totals(foo)) %>%
    transpose() %>%
    row_to_names(row_number=1)
```

With the `airbus_han_ids`, we will extract patents from `HAN_PATENTS` dataset. Then we found 1219 patent documents which are applied in 
different countries. When we use patent counting to measure how innovative of a firm is, we should be aware of the 
issue of over counting {% cite webb2005analysing %}. 



<div class="table-caption">
<span class="table-caption-label">Table 9.</span> Patent distribution for Airbus Defence (DE)
</div>

Office | EP  | US  | WO | total
|:---:|:---:|:--:|:---:|
Number | 716 | 415 | 88 | 1219


__Over counted or under counted?__ In total, we got 1219 patent documents from different patent offices.
However, the same patent could be filed in different offices. In our 
case, we need to check whether there are duplicates for our entity -
_AIRBUS DEFENCE AND SPACE GMBH_. For instance, among 1219 patents,
document `US2017113777` refers to the patent application document
that the firm filed in USPTO whereas document `EP3162699` refers to
the __same__ patent application file and granted patent document by EPO. However,
both patent documents are recorded in `HAN_PATENTS` dataset. 

<div class="table-caption">
<span class="table-caption-label">Table 10.</span> Some of duplicated patents 
</div>

| HAN_ID | HARM_ID | Appln_id  | Publn_auth |Patent_number | Search_link|
|:------:|:-------:|:---------:|:----------:|:-------------:|:----------:|
| 60513  |  60513  | 470692725 |     EP     |EP3162699     |[Espacenet](https://worldwide.espacenet.com/searchResults?query=EP+3162699+B1){:target="_blank"}
| 60513  |  60513  | 477856200 |     US     |US2017113777  |[Espacenet](https://worldwide.espacenet.com/searchResults?ST=singleline&locale=en_EP&submitted=true&DB=&query=US2017113777%0D%0A){:target="_blank"}

Table 10 only shows two of those duplicates, there might be more. However,
it is not practical to go through all those patent numbers and search them
one by one and then check whether there are some duplicates or not. For now,
we will only use patents from EPO. This should be a safe choice as EPO
normally grant less patents with stricter rules (see group 1 in Table 11). 

<div class="table-caption">
<span class="table-caption-label">Table 11.</span> Some of duplicated patents with kind code 
</div>

| HAN_ID  | Publn_auth |Patent_number | Kind_code| Group |
|:------:|:----------:|:-------------:|:-------------|:-------------:|
| 60513   |     US     |US9308691     | B2 (granted) | 1 |
| 60513  |    EP     |EP2689872     | A3 (application+search report) | 1| 
| 60513   |     US     |US10703486    | B2 (granted) | 2| 
| 60513    |      EP     |EP3219614     | B1 (granted) | 2| 
| 60513  |     US     |US2017165795  | A1 (application)| 3 | 
| Not in the database |     EP     |EP3181711| B1 (granted)| 3| 


For our patent analysis, we need to make sure:
- within one patent office (say EPO):
    - patents are not over counted
    - patents are not under counted
- across patent offices (say USPTO and EPO):
    - patents are not over counted
    - patents are not under counted

If we only use patents from EPO, patents might be under counted overall as it
is possible that firms got granted patents from USPTO but did not filed patent application to EPO or applied but rejected. With all those trade-offs, the best decision 
is to be of under counted rather than over counted. 

In Table 11, there is one entry marked as 'Not in the database', this shows
the bias[^4] of OECD HAN database because `EP3181711` was granted by EPO and also
published as `US2017165795`. However, only one of them is included in the
database as shown in Table 11. 

[^4]: it is unavoidable for having bias when you work with millions of documents; as long as it is not systematic it should be okay. 


__Using EPO patents.__ By using EPO patent numbers, we can avoid the issue of
over count across patent offices with the risk of under count. However,
we have not solved the issue of over count within EPO. Table 12 shows not all patents from `HAN_PATENTS` were granted. 

<div class="table-caption">
<span class="table-caption-label">Table 12.</span> Selected patents for Airbus defence (DE) from EPO 
</div>

| HAN_ID | HARM_ID | Appln_id | Publn_auth | Patent_number | Granted |
|:------:|:-------:|:--------:|:----------:|:-------------:|:-------------:|
| 60513  |  60513  |   213    |     EP     |   EP2030891   | No |
| 60513  |  60513  |  65448   |     EP     |   EP2025928   | No |
| 60513  |  60513  |  156990  |     EP     |   EP1920908   | Yes |
| 60513  |  60513  |  161551  |     EP     |   EP1972896   | Yes |
| 60513  |  60513  |  173385  |     EP     |   EP2134522   | Yes |
| 60513  |  60513  |  173386  |     EP     |   EP2136979   | Yes |

__Summary.__ At this stage, let's summarize what have done. 

```r
# Code-Block 5. Summary of analysis until now
# read dataset 1, 2, and 3 
de_firms <- fread('work/notebooks/patent/data/orbis_de_matched_l.csv')
han_names <- fread('work/notebooks/patent/data/202208_HAN_NAMES.txt')
han_patents <- fread('work/notebooks/patent/data/202208_HAN_PATENTS.txt')

# filter out germany firms from han_names 
# by setting Person_ctry_code == 'DE'
# match names "AIRBUS DEFENCE" and get their HAN_ID
airbus <- toupper('Airbus Defence')
han_names %>%
    .[Person_ctry_code == 'DE'] %>%
    .[Clean_name %like% airbus] %>%
    .[,HAN_ID] -> airbus_han_ids

# calcualte the summary statistics for AIRBUS DEFENCE
han_patents %>%
    .[HAN_ID %in% airbus_han_ids] %>%
    .[, .N, by=Publn_auth] -> foo
    transform(adorn_totals(foo)) %>%
    transpose() %>%
    row_to_names(row_number=1)

# focusing on patents from EPO
# filter with condition Publn_auth == 'EP'
han_patents %>%
    .[HAN_ID %in% airbus_han_ids] %>%
    .[Publn_auth == 'EP'] -> airbus_ep_patents
```

With all patent documents of AIRBUS DEFENCE AND SPACE GMBH from EPO we 
ready to figure out how to search for granted patents. The input for 
the next step should be `airbus_ep_patents`, which is a table with 
$716$ rows as shown in Table 12. 

### Search for granted patents 

If you have registered PATSTAT online database (free trial for one month) or 
bought the database, you can extract more information from it based on
`Appln_id` which is the unique identifier used in PATSTAT. For instance,

```sql
SELECT * FROM tls211_pat_publn  -- tls211_pat_publn is the publication table
WHERE appln_id = 156990  -- third row from Table 12
```
With the above query, I got the following results. 

| pat_publn_id | publn_kind | appln_id | publn_date | publn_first_grant |
|:------------:|:----------:|:--------:|:----------:|:-----------------:|
|  277148936   |     A1     |  156990  | 2008-05-14 |         N         |
|  437725522   |     B1     |  156990  | 2015-04-08 |         Y         |

This tells us whether the patent was granted or not. However, we need to
extract this kind of information with thousands of entries (if not millions).
Right now, I could not find a way from PATSTAT online to do this (please email
to me if you knew how to do it, thanks in advance). 

To search for the granted information in a programming environment,
we need an API. Luckily, EPO provides this kind of APIs:

- one is  called
[Open Patent Services](https://www.epo.org/searching-for-patents/data/web-services/ops_de.html){:target="_blank"}
    - use this when you want to search or query from scratch
    - use this when you want to extract information from major patent office
- the other one is called [Linked open EP data](https://www.epo.org/searching-for-patents/data/linked-open-data.html)
    - use this when you already have patent numbers with `EP` format
    - for non-EP documents (such as `US10703486`), coverage is limited to basic patent identification information
    - there is no search API endpoint from this open service, but one can search or query with [SPARQL](https://en.wikipedia.org/wiki/SPARQL){:target="_blank"} if you knew SPARQL well (it is not recommend using
    it as it is not very stable when I test it).  

Since we use __only EPO patents__ and  have patent numbers and application IDs from OECD HAN database, we will use Linked open EP data. 

### Linked open EP data 

According to EPO, "Linked open data offers you new ways of combining
patent data and non-patent data in your work". Linked open EP data can be queried, retrieved and viewed using standardized web technologies like HTTP,
URI and [SPARQL](https://en.wikipedia.org/wiki/SPARQL){:target="_blank"}.

__Back to our case.__ As we have stated from the last section, we want to
search for granted patents based on `Patent_number`. Those patent numbers we 
have are publication numbers. Therefore, we should use APIs for publications.
Please read the following documents from EPO:

- [Coverage of linked open EP data](https://data.epo.org/linked-data/documentation/data-coverage.html){:target="_blank"}
- [API Overview](https://data.epo.org/linked-data/documentation/api-overview.html){:target="_blank"}
- [API Reference](https://data.epo.org/linked-data/documentation/api-reference.html){:target="_blank"}

The idea behind API is very intuitive. For each URL, you have an endpoint
which returns the specific results. For instance, the following link returns
all patents with publication numbers as input:

```
https://data.epo.org/linked-data/data/publication/{st3Code}/{publicationNumber}

# st3Code is two-letter codes of countries 
# publicaionNumber is just publication number

# try this one in your browser 
# https://data.epo.org/linked-data/data/publication/EP/1048543
```

With publication number (`Panten_number` in our database), we can just
extract all relevant information from linked database of EPO. Code-Block 6
gives the sample code in R to extract all publications for patent EP1972896.

{% endkatexmm %}

```r
# Code-Block 6. API request
request <- GET('https://data.epo.org/linked-data/data/publication/EP/1972896.json')
response <- content(request, as = "text", encoding = "UTF-8")
json<- fromJSON(response, flatten = TRUE)

# make a table with selected variables
json$result$items %>%
    select(
        `_about`, 
        publicationDate, 
        application.applicationNumber,
        publicationKind.label) %>%
    rename(Link=`_about`, Publication_date=publicationDate, 
        Application_number=application.applicationNumber, 
        Kind_code=publicationKind.label)
```

Table 13 gives the results of Code-Block 6, which shows the key information
we care: whether the patent is granted or not, which is indicated by
the kind code. The rule of finding granted patents is to filter 
out `B1` or `B2` from kind code. 

<div class="table-caption">
<span class="table-caption-label">Table 13.</span> Returned results from
EPO's publication API
</div>

|Link                                                             | Publication_date | Application_number | Kind_code |
|:----------------------------------------------------------------|:----------------:|:------------------:|:---------:|
|[EP/1972896/A2/](http://data.epo.org/linked-data/data/publication/EP/1972896/A2/-){:target="_blank"} | Wed, 24 Sep 2008 |      08004318      |    A2     |
|[EP/1972896/A3/-](http://data.epo.org/linked-data/data/publication/EP/1972896/A3/-){:target="_blank"} | Wed, 07 Nov 2012 |      08004318      |    A3     |
|[EP/1972896/B1/-](http://data.epo.org/linked-data/data/publication/EP/1972896/B1/-){:target="_blank"}  | Wed, 06 May 2015 |      08004318      |    B1     |

With the right algorithm as shown in [Code-Block 7](https://gist.github.com/oceanumeric/51ab6c7f28b3df244bf6d773c35d003f){:target="_blank"}, we can extract 
all patent information for `AIRBUS DEFENCE AND SPACE GMBH`. To extract patent information for 716 patent numbers of Airbus company, it
takes around 6 minutes and 38 seconds. It could be speeded up if
we set a smaller value of `Sys.sleep()`. 


<div class='figure'>
    <img src="/images/blog/airbus_summary.webp"
         alt="Airbus patents distribution"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Summary of granted
        patents of Airbus Defence; the decline in recent years can be explained with the fact that the application and granting procedures may
take several years before a patent is granted, similar to many other fields of technology.
    </div>
</div>

With 538 granted patents of
Airbus Defence and Space company, we will search for their classifications
and try to find out the field of strategic orientation of this firm. 

__International Patent Classification (IPC).__ The International Patent Classification (IPC) system
is a hierarchical patent classification system which is
used by more than 100 patent offices on all continents.
It breaks down technologies into eight sections with
several hierarchical sub-levels. The IPC system has
approximately 75,000 subdivisions and is updated on
an annual basis. Further information on the IPC system
is available at [WIPO's IPC Guide](https://ipcpub.wipo.int/?notion=scheme&version=20230101&symbol=none&menulang=en&lang=en&viewmode=f&fipcpc=no&showdeleted=yes&indexes=no&headings=yes&notes=yes&direction=o2n&initial=A&cwid=none&tree=no&searchmode=smart){:target="_blank"}.

__Cooperative Patent Classification (CPC).__ The Cooperative Patent Classification (CPC) system
builds on the IPC system and provides a more granular
and detailed classification structure. The CPC system
has more than 250,000 subdivisions and is updated four
times a year. It is used by more than 30 patent offices
worldwide. Detailed information about the CPC system
is available at [USPTO's CPC Lookup page](https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html){:target="_blank"}.

With the program in [Code-Block 8](https://gist.github.com/oceanumeric/1a23ef7f8cd99653eb26f6d7882023ed){:target="_blank"}, we will calculate
how many IPC classes that Airbus Defence (DE) applied for and what 
are the most frequent ones in terms of level 1, 2 and 3. For IPC,
level 1 classification has 8 categories:

- A: Human Necessities
- B: Performing Operations, Transporting
- C: Chemistry, Metallurgy
- D: Textiles, Paper
- E: Fixed Constructions
- F: Mechanical Engineering, Lighting, Heating, Weapons
- G: Physics
- H: Electricity

Level 2 of IPC gives more details of top level, for instance, `B06` is 
everything about cleaning and `H04` is about electronic communication
technique. Within each sub-level, you have sub-level too. Let's go through
a case with IPC number _H04H 60/18_ on copying information:

- H04H: BROADCAST COMMUNICATION 
    - H04H 60: 	Arrangements for broadcast applications with a direct linkage to broadcast information or to broadcast space-time; Broadcast-related systems
        - H04H 60/09: Arrangements for device control with a direct linkage to broadcast information or to broadcast space-time; Arrangements for control of broadcast-related services
            - H04H 60/14: Arrangements for conditional access to broadcast information or to broadcast-related services
                - H04H 60/18: on copying information 


Where could we get the information of IPC or CPC symbols without looking
up one by one? The answer is again [Linked Open EP data](https://data.epo.org/linked-data/documentation/api-reference.html){:target=":_blank"}. 

<div class='figure'>
    <img src="/images/blog/airbus_ipc_top.png"
         alt="Airbus patents distribution"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The IPC distribution of
        Airbus Defence (DE)'s patents; section B is about performing operations and transport; section H is electricity; section G is about
        physics.
    </div>
</div>