---
title: R data.table Best Practices
subtitle: There are many packages in R or Python to deal with table like data (or so called data frame), but data.table is probably the most efficient one. Here are some best practices to use data.table.
layout: blog_default
date: 2023-04-23
keywords: data-science, data-analysis, data-table, r-programming
published: true
tags: data-science, r 
---

I have been planning to write a post on `data.table` for a while. The motivation is that
many posts on data science or data analysis in R are using `data.frame` or `dplyr` to
manipulate data. Many people probably will go for `pandas` in Python in the first place.
However, I appreciate the efficiency of `data.table` more and more now, and believe for the dataset with is larger than 1GB and less than 100GB, `data.table` is the best choice.

After many years of using both `Python` and `R`, I have found that `data.table` is the most efficient package to deal with table-like data. It is also the most efficient package to deal with large dataset in `R`. 

I also realized that there are many posts on the internet teaching people how to use all
kinds of packages in `R`, especially `tidyverse`. First of all, I have to say
that I like `dtidyverse` a lot, especially `ggplot2` and `dplyr`. However, I think
using `data.table` with basic `R` syntax is more efficient than using different kinds 
of packages. I believe using `data.table` with basic `R` syntax very well is the best
way to learn `R` for data analysis.


So, here are the messages I have learned from my experience:


- Use `Excel` for small dataset, such as less than 100MB (it is probably already too large for `Excel`).
- Use `data.table` with basic `R` syntax for any dataset in the format of table-like or data-frame between 100MB and 100GB.
- Use `Python` for any dataset that is not structured in the format of table-like or data-frame.
- Use basic `R` syntax for data visualization and only use `ggplot2` when you need to do some advanced visualization. 
- Use `Python` for machine learning and deep learning projects.
- Use `R` for statistical analysis and data analysis projects in finance, economics, and other fields.
- Keep trying different packages and learn from others, such as `Polars`.

If you do not believe me why I think `data.table` is the efficient one, please  check the [benchmark](https://duckdblabs.github.io/db-benchmark/){:target="_blank"} of different packages on data manipulation.

Here is the roadmap of this post:

- [Big picture](#big-picture)
- [Best practices](#best-practices)


## Big picture

Broadly speaking, there are there kinds of tasks in data analysis:

1. what is the data about? such as the data structure, the data type, the data size, and variable names, etc.
2. the data manipulation, such as data cleaning, data transformation, data aggregation, and data merging, etc.
3. the data analysis, such as data visualization, statistical analysis, and machine learning, etc.

I will use a small dataset to illustrate the best practices for each task. One could use those best practices to deal with any dataset in the format of table-like or data-frame
even if the dataset is very large (e.g., 100GB).

All code in this post uses `%>%` from `magrittr` package heavily, which is a very useful package to chain functions together. The pipe operator `%>%` in `R` community is one of the most important features in `R`, which makes me to favor `R` over `Python` in data analysis for table-like data.

Before we start, you need to understand that table-like data is a two dimensional data structure, which is also called data frame in `R` or `Python`. The first dimension is the row dimension, and the second dimension is the column dimension. The row dimension is the observation dimension, and the column dimension is the variable dimension. Figure 
1 illustrates how `data.table` manupulates data in the row dimension and the column dimension with the syntax of `i` and `j`.

<div class='figure'>
    <img src="/images/blog/R-data-table-illustration.png"
         alt="fp7 totalcost hist1" class="zoom-img"
         style="width: 76%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The illustration of LDA from the paper by David Blei (2012). You can click on the image to zoom in.
    </div>
</div>

## Best practices

Let's import key packages and load the dataset. When you do a project, it is better 
to use as less packages as possible because it is easier to debug and maintain the code.
If you have too many packages, the dependencies of the packages may conflict with each other and it also makes the code harder to maintain and debug.

```R
library(data.table)
library(magrittr)
library(knitr)  # for kable, a function to print a table in different formats
library(ggplot2)  # only use it for advanced visualization
# for dplyr, it is recommended to call it as dplyr::filter, dplyr::select, etc.
```

The dataset we will use is based on a survey I did for one of my tutorials. Here is
the [link](https://docs.google.com/forms/d/1VoDT0dknxvx1T_RDILHOYvLH226ZCFMhkUuRtJvyT60/edit){:target="_blank"} to the survey.
It has 10 questions and the answers are in the format of single choice, multiple 
choices, and open answers. The dataset is in the format of table-like data, which is
stored in a [csv file](https://raw.githubusercontent.com/oceanumeric/data-science-go-small/main/Tutorial_01/survey_responses.csv){:target="_blank"}. I added duplicated rows to the dataset as an example of how to deal with duplicated rows.


### structure of the data: the first step 

Every time I got a new dataset, I use three functions to check the structure of the data: `head`, `str`, and `summary`. The `head` function prints the first 6 rows of the dataset. The `str` function prints the structure of the dataset, which includes the data type of each variable. The `summary` function prints the summary statistics of each variable. 

```R
# read data
dt <- fread("https://raw.githubusercontent.com/oceanumeric/data-science-go-small/main/Tutorial_01/survey_responses.csv")


# data structure
str(dt)  # 24 observations of 11 variables
head(dt)  # check the first 6 rows
summary(dt)
```

### missing values are always tricky

To check how many missing values are in the dataset, we can use the `is.na` function. The `is.na` function returns a logical vector with the same length as the dataset. Combining the `is.na` function with `sum` and `sapply` functions, we can check how many missing values are in each variable. 

```R
########## --------- check missing values --------- ###########

# Missing values are tricky as they can be represented in different ways
# such as "", "Na", "NA", "na", "N/A", "n/a", "NA/NaN", "NA/NaN/Na", etc.

# check missing values and print it as a table
sapply(dt, function(x) sum(is.na(x))) %>% kable()

# there is only one missing value in q2?
sapply(dt, function(x) sum(x =="")) %>% kable()
sapply(dt, function(x) sum(x == "Na")) %>% kable()

sapply(dt, function(x) sum(x =="", na.rm = TRUE)) %>% kable()
sapply(dt, function(x) sum(x == "Na", na.rm = TRUE)) %>% kable()

# replace missing values (represented as strings) with NA
dt[dt == ""] <- NA
dt[dt == "Na"] <- NA

sapply(dt, function(x) sum(is.na(x))) %>% kable()
```

### column transformation

As it is shown in Figure 1, all column operations are done in the `j` part of the `data.table` syntax.

The `data.table` package provides a very efficient way to manipulate data in the column dimension. The `:=` operator is used to create a new variable. The `:=` operator is very similar to the `=` operator in `R`. The difference is that the `:=` operator does not create a new copy of the dataset. The `:=` operator modifies the dataset in place, which means doing operations on the original dataset. 

If you do not want to modify the original dataset, you can use `.()` to create a new copy of the dataset. One could also use `c(column_name)` to create a new copy of the dataset. 

In summary, here are ways to select and manipulate columns in `data.table`:

- return a new copy of the dataset: 
    - `dt[, .(variable_name1, variable_name2)]`
    - `dt[, c(1, 2, 3:7)]` select the first, second, and the third to the seventh columns
- return a vector of the column: 
    - `dt[, variable_name]`
    - `dt %>% .[, (variable_name)]` the dot `.` represents the dataset

- Create new variable in place (on original dataset)
    - `dt[, new_variable_name := expression]`
    - `dt %>% .[, new_variable_name := expression]` the dot `.` represents the dataset

- Create new variable in a new copy of the dataset
    - `dt[, .(new_variable_name = expression)]`
    - `dt %>% .[, .(new_variable_name = expression)]` the dot `.` represents the dataset

```R
########## --------- column operations --------- ###########

# check structure of dt again
str(dt)

# convert timeStamp to date format by adding a new column
# called dateTime to dt
dt %>%
    .[, dateTime := as.POSIXct(timesStamp, format = "%m/%d/%Y %H:%M:%S")] %>%
    str()

# create a new column called year
dt %>%
    .[, year := format(dateTime, "%Y")] %>%
    # you can also convert it to numeric
    # .[, year := as.numeric(format(dateTime, "%Y"))] %>%
    str()

# select columns and return a new data.table
dt %>%
    .[, .(year, q2, q3)] %>%
    str()

# select q9: 9. Did you try ChatGPT? 
# convert it to lower case
dt %>%
    .[, .(q9 = tolower(q9))] %>% 
    str()

# select columns year, and from q3 to q10
# one needs to know the column index
dt %>%
    .[, c(13, 4:11)] %>%
    str()

# return a vector
dt %>%
    .[, (q7)]
```

Sometimes, we want to do operations on multiple columns based on their names. 
For instance, we want to select all columns with names starting with `q`. For this,
we can use `.SDcols` to select columns based on their names. The `.SDcols` is a
special variable in `data.table` that represents the selected columns. `SD` stands
for Subset of Data.

By combing `.SDcols` with `.SD`, we can:

- select columns based on their names
    - `dt[, .SD, .SDcols = c("q1", "q2", "q3")]`
    - `dt[, .SD, .SDcols = c(1, 2, 3)]`
    - `dt[, .SD, .SDcols = patterns("^q")]` select columns with names starting with "q"
- do operations on the selected columns
    - `dt[, lapply(.SD, mean), .SDcols = c("q1", "q2", "q3")]` compute the mean of columns q1, q2, and q3
    - `dt[, lapply(.SD, tolower), .SDcols = c(1, 2, 3)]` convert all columns to lower case
    - `dt[, lapply(.SD, mean), .SDcols = patterns("^q")]`

I provide four examples below to show how to select columns based on different criteria.

```R
# select columns with names starting with "q"
dt %>%
    .[, .SD, .SDcols = grep("^q", names(dt))] %>%
    str()

# or using patterns to select columns
dt %>%
    .[, .SD, .SDcols = patterns("^q")] %>%
    str()

# we can also do operations on multiple columns

# select columns based on their data types == int
dt %>%
    .[, .SD, .SDcols = is.integer] %>% str()

# convert values from q2 to q10 to lower case if they are characters
# return a new data.table
dt %>%
    .[, .SD, .SDcols = is.character] %>%
    .[, lapply(.SD, tolower), .SDcols = patterns("^q")] %>%
    str()
```

I believe the above code covers most of the common operations in data transformation in terms of selecting columns. For columns, we have two
properties: _names and data types_. We can select columns based on their names or data types, or both by using `%>%` to chain multiple operations.

### row transformation

When we talk about row operations, we have two properties: _row indices and row values_. Based on those two properties, we can:

- operate on _row values_ without any row indices, which means passing
the all values in one row to a function.
- operate on _row values_ with row indices, which means
    - we filter the dataset based on some criteria
    - we subset the dataset based on row indices or some criteria

When we do operations on row values, we will use `with` heavily in the pipeline as it could help us to do:

1. call functions on the row values within pipelines
2. call basic plotting functions on the row values within pipelines

```R
# one row operation, without any row indices
# summarize q1 - chr, yes, no
# treat it as factor, using table() to count the number of each level

table(dt$q1)

# or convert it to factor first
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    table() 

# better presentation
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    .[, .(count = table(q1))] %>%
    kable()

# calculate the percentage of each level
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    .[, .(count = table(q1))] %>%
    .[, share := count.N / sum(count.N)] %>%
    kable()

# plot it as a bar chart
# set options for the size
options(repr.plot.width = 8, repr.plot.height = 5)
dt %>%
    with(table(q1)) %>%
    barplot(main = "Did you learn regression model before?")

# plot share instead of count
dt %>%
    with(table(q1)/nrow(dt)) %>%
    barplot(main = "Did you learn regression model before?")


# more advanced visualization with ggplot2
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    .[, .(count = table(q1))] %>%
    .[, share := count.N / sum(count.N)] %>%
    ggplot(aes(x = count.q1, y = share)) +
    geom_col(fill = "#6F6CAE") +
    geom_text(aes(label = round(share, 2)), vjust = -0.5) +
    labs(title = "Did you learn regression model before?")
```