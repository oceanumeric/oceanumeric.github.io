---
title: Deal With Duplicates in R
subtitle: Inspecting big data could be challenge in terms of finding duplicates and dropping them.
layout: blog_default
date: 2023-03-05
keywords: data science, Python, R, big data, duplicates, unique values
published: false
tags: r python data-science big-data
---

Finding unique and duplicate values in a large dataset happens very often for people who are dealing with big data. 


Although the `unique()` function in R could give us the unique
values in any dataset. However, sometimes, we want to inspect those
duplicates with our eyes. How could we do this? 


```R
dim(ipc1)  # 130803

head(ipc1)

# A data.table: 6 × 5
# V1	patentNumber	familyID	kindCode	pubAuth
# <int>	    <chr>	  <int>	    <chr>	   <chr>
# 0	US2010005485	38188959	A1	US
# 1	WO2010005798	41506050	A2	WO
# 2	WO2010005270	41507602	A2	WO
# 3	WO2010004958	41507071	A1	WO
# 4	WO2010004719	41506847	A1	WO
# 5	US2010010995	40104654	A1	US

ipc1 %>%
    unique(by = "patentNumber") %>%
    dim()  # 120394 

130803 - 120394
# 10409

# inspect duplicates
n_occur <- data.table(table(ipc1$patentNumber))

n_occur %>%
    .[N > 1] %>%
    head()

# A data.table: 6 × 2
# V1	        N
# <chr>	      <int>
# EP2162828	    2
# EP2162838	    2
# EP2162853	    2
# EP2162862	    2
# EP2163094	    2
# EP2163183	    2

n_occur %>%
    .[N > 2] %>%
    head()

# A data.table: 6 × 2
# V1	        N
# <chr>	      <int>
# EP3256964	    3
# EP3256992	    3
# EP3256993	    3
# EP3257002	    3
# EP3257019	    3
# EP3257020	    3


n_occur %>%
    .[N > 1] %>%
    dim()  # 10063 


n_occur %>%
    .[N > 1] %>%
    .[, .(total = sum(N))]  # 20472

20472 - 10063  # 10409
```

It is normal that some values occur more than one times (or twice). 