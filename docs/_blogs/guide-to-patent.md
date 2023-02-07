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

Figure 2 gives OECD HAN database structure (open the zoomed version 
with the new tab by right-click). It has four tables: `HAN_PERSON`, `HAN_NAMES`,
`HARM_NAMES`, and `HAN_PATENTS`. We will use one patent file as an example to
walk through those tables. Notice that there are around 18 million rows in
`HAN_PATENTS` table, which accounts for around 18% of EPO's total patent document.[^1]
This means the OECD HAN database has a European-centered bias. This is partly
due to the ORBIS database which has more register information for European firms.
We will discuss how to get and analyze patent data for Chinese firms in the
coming sections of this post. 


[^1]:there are more than 100 million patent documents from leading industrialized and developing countries in EPO's database.



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

Now, we chose the first six records (or entries) from `HAN_PATENTS` table and search
those patents from EPO's [Espacenet](https://worldwide.espacenet.com/?locale=en_EP).
In the `HAN_PATENTS` table, the variable `Appln_id` refers to the patent application
identifier in PATSTAT database. You can use it to retreat the document if you
can access to PATSTAT database. When we search in Espacenet, we use `Patent_number`
instead of `Appln_id`. 

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
        <span class='caption-label'>Figure 3.</span> Espacenet's searching
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
at the Figure 4. 

<div class='figure'>
    <img src="/images/blog/patent-searching-results.png"
         alt="Searching results illustated for US8668089"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The complexity of
        a patent file. 
    </div>
</div>


