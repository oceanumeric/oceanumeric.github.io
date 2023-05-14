---
title:  Everything that exists is an object in R 
subtitle: According to John Chambers, everything that exists is an object in R and everything that happens is a function call. This summarizes the philosophy of R. Knowing this, we can understand R better.
layout: blog_default
date: 2023-05-14
keywords: data science, python, R, VS code, stack tools, IDE
published: true
tags: data-science r vs-code 
---

The author of R, John Chambers, said that everything that exists is an object in R and everything that happens is a function call. This summarizes the philosophy of R. Knowing this, we can understand R better.

I came across this statement when I notice the following code:

```R
print(labels_str)
# keine Stutzung       Stutzung 
#              0              1 
typeof(labels_str)
# double
as.character(labels_str)
# "0" "1"
```

As a `Python` user, I was surprised by the output of `typeof(labels_str)`. I thought it should be `character` instead of `double`. Then I found the following explanation from [Stackoverflow](https://stackoverflow.com/questions/34376318/whats-the-real-meaning-about-everything-that-exists-is-an-object-in-r){:target="_blank"}. 


Unlike Python, R is a functional programming language. This means that everything that happens is a function call. This is the reason why you will never see the
following code in `R`:

```python
df.head()
```

Instead, you will see:

```R
head(df)
```

Unlike Python, everything that exists is an object in R. This means you can do the following:

```R
obj <- 1:3
names(obj) <- c("a", "b", "c")
print(obj)
# a b c
# 1 2 3
```

You can never do this in Python. That's why in `R`, you can do the following:

```R
is.double(labels_str)
# TRUE
print(labels_str)
# keine Stutzung       Stutzung
#              0              1
as.data.frame(labels_str)
#                labels_str
# keine Stutzung          0
# Stutzung                1
```

The attributes of a variable can be queried with `attributes` and `str`.


Specially treated attributes (attribute names) are:

- `class` (a character vector with the classes that an object inherits from).
comment
- `dim` (which is used to implement arrays)
- `dimnames`
- `names` (to label the elements of a vector or a list).
- `row.names`
- `tsp` (to store parameters of time series, start, stop and frequencies)
- `levels` (for factors)


The class in `R` is very subtle. One could understand it as attributes of an object with flexible types. To learn more this, one could read [Advanced R](https://adv-r.hadley.nz/oo.html){:target="_blank"}.


## An example

```R
stata_data <- haven::read_dta("../data/innovation_survey/extmidp21.dta",
                                encoding = "windows-1252")

str(stata_data)
# tibble [5,083 × 284] (S3: tbl_df/tbl/data.frame)
#  $ id          : chr [1:5083] "300127" "301003" "301078" "301084" ...
#   ..- attr(*, "label")= chr "Identifikation externe"
#   ..- attr(*, "format.stata")= chr "%10s"
#  $ branche     : dbl+lbl [1:5083]  9,  8, 10,  1,  1,  9,  1,  6,  9,  1,  2, 10,  9,  ...
#    ..@ label       : chr "Einteil. in 21 Wirtschaftszweige"
#    ..@ format.stata: chr "%38.0g"
#    ..@ labels      : Named num [1:21] 1 2 3 4 5 6 7 8 9 10 ...
sapply(stata_data, attributes) %>% 
    as.data.table() %>%
    # add one row with the variable names
    rbind(as.list(names(stata_data))) %>%
    # transpose the data.table
    t() %>%
    # convert to data.table
    as.data.table() %>%
    # select columns
    .[, c(5, 1, 3, 4)] %>%
    # set names
    setnames(c("variable", "labels", "type", "values")) %>%
    kable()
```