---
title: CORDIS - EU Research Projects Datasets
subtitle: The dataset contains all projects funded by the European Union from 1998 to 2027, I organized those datasets in this post for research purpose. 
layout: blog_default
date: 2023-02-19
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



## Horizon 2020 dataset 


<div class="table-caption">
<span class="table-caption-label">Table 1.</span> Distribution of funded projects in terms of industry
</div>


|                                Industry                                 |N    |
|:-----------------------------------------------------------------------|:----:|
|                    computer and information sciences                    |4525 |
|                           biological sciences                           |4454 |
|                            physical sciences                            |2335 |
|                        environmental engineering                        |2105 |
| electronic engineering, information engineering |1784 |
|                            chemical sciences                            |1739 |
|                                sociology                                |1625 |
|                            clinical medicine                            |1603 |
|                         economics and business                          |1517 |
|                             basic medicine                              |1340 |


```r
# load packages
library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    archive, kableExtra, SPARQL, janitor,
    png, webp, Cairo, rsvg, rbenchmark,
    httr, jsonlite, fedmatch)
# set option
options(dplyr.summarise.inform = FALSE)
options(repr.plot.width = 6, repr.plot.height = 4)

gray_scale <- c("#F3F4F8", "#D2D4DA",  "#bcc0ca",
                "#D3D3D3", "#2B2B2B",
                "#B3B5BD", "#838383",
                "#9496A1", "#7d7f89", "#777986",
                "#656673", "#5B5D6B", "#4d505e",
                "#404352", "#2b2d3b", "#282A3A",
                "#1b1c2a", "#191a2b", "#141626", "#101223")

# check working directory
getwd()


########## ------- Create a merged dataset ----------- ###############
projects <- fread("./data/cordis_h2020/project.csv")
org <- fread("./data/cordis_h2020/organization.csv")
sv_tags <- fread("./data/cordis_h2020/euroSciVoc.csv")

# transform projects by selecting some columns
names(projects)
names(projects)[1] <- "projectID"

projects %>%
    .[, startYear := as.numeric(format(startDate, "%Y"))] %>%
    .[, endYear := as.numeric(format(endDate, "%Y"))] %>%
    .[, c(
        "projectID", "startYear", "endYear",
        "status", "title", "topics",
        "fundingScheme", "grantDoi")] -> projects_subset


# view organization
org %>%
    .[sample(.N, 3)]


# view sv_tas and # extract topics
sv_tags %>%
    .[, c(1:4)] %>%
    .[, svLevel1 := tstrsplit(
        euroSciVocPath, "/", fixed = TRUE, keep = c(2))] %>%
    .[, svLevel2 := tstrsplit(
        euroSciVocPath, "/", fixed = TRUE, keep = c(3))] -> sv_tags_levels

# group by title
sv_tags_levels %>%
    .[, c(1, 4)] %>%
    .[, lapply(.SD, paste0,  collapse = "/"), by = projectID] -> sv_tags_title

# merge and get unique
sv_tags_title %>%
    .[sv_tags_levels, on = .(projectID)] %>%
    unique(by = "projectID") %>%
    .[, c(1:4, 6, 7)] -> sv_tags_sum


# merge projects_sub, org and sv_tags_levels 
projects_subset <- merge(projects_subset, sv_tags_sum,
                         by = "projectID", all.x = TRUE)
horizon <- merge(org, projects_subset, by = "projectID", all.x = TRUE)


# on projects level
projects_subset %>%
    .[, .(svLevel2)] %>%
    .[, .N, by=svLevel2] %>%
    .[order(-rank(N))] %>%
    head(10) %>%
    kable("pipe", align = "cl")

# one project is normally applied by many organizations 
horizon %>%
    .[sample(.N, 5)]

names(horizon)


# filter out German firms
horizon %>%
    .[, c(1, 2, 3, 5, 7, 8:12, 23, 26, 27, 32, 33, 37)] %>%
    .[, totalCost := gsub(",", ".", totalCost)] %>%
    .[, totalCost := as.numeric(totalCost)] %>%
    .[country == "DE" & activityType == "PRC"] -> horizon_de_firms


dim(horizon_de_firms)
names(horizon_de_firms)
horizon_de_firms[sample(.N, 5)]



########## ------ Match names with Orbis dataset ------ #########
orbis_de <- fread("./data/orbis_de_firms.csv")

orbis_de %>%
    .[, .(bvdid, name_internat, category_of_company,
                city_internat, postcode)] %>%
    # create another ID
    .[, orbisID := .I] %>%
    .[, cleaned_names := clean_strings(name_internat)] -> foo1

names(foo1)

horizon_de_firms %>%
    .[, horizonID := .I] %>% 
    .[, cleaned_names := clean_strings(name)] -> foo2

names(foo2)


# match names
merge_foo <- merge_plus(data1 = foo1, data2 = foo2,
                        by.x = "cleaned_names", by.y = "cleaned_names",
                        match_type = "fuzzy",
                        unique_key_1 = "orbisID",
                        unique_key_2 = "horizonID")
```