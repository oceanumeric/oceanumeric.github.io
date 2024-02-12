---
title: R data.table from an advanced standpoint
subtitle: The more you use it, the more you love it
layout: blog_default
date: 2024-02-12
keywords: data-science, data-analysis, data-table, r-programming, python, pandas, sql, duckdb
published: true
tags: data-science r python data-table pandas sql duckdb
---

Whenever I do data analysis, I always start from `data.table` in R. It is a powerful package that is designed to handle large datasets. If I encountered a problem that I cannot solve with `data.table`, I would then switch to `pandas` in Python. `pandas` or `polars` from python ecosystem are very handy when it comes to processing json data or when I need to use some third-party libraries that are not available in R, such as translation or image processing.

Recently, I have discovered `duckdb` which is a new database engine that could allow user to run SQL queries on big data without the need to install a database server. I see the huge potential of `duckdb` especially when it can be used as a complement to `data.table` and `pandas`. 

In this blog post, I will try to summarize what I have learned about `data.table` and present a framework from an advanced standpoint. With this framework, you will have a better understanding of how to use `data.table` to solve complex problems and how to think like a data scientist and software engineer at the same time.

- [Advanced standpoint 1: it is all about the object](#advanced-standpoint-1-it-is-all-about-the-object)
- [Advanced standpoint 2: programming = data structure + algorithm](#advanced-standpoint-2-programming--data-structure--algorithm)
- [data.table from an advanced standpoint](#data-table-from-an-advanced-standpoint)
  - [Checking missing values for each column](#checking-missing-values-for-each-column)
  - [Checking the summary statistics for selected columns](#checking-the-summary-statistics-for-selected-columns)
  - [Order the count within each group](#order-the-count-within-each-group)
  - [Working with nested data](#working-with-nested-data)


## Advanced standpoint 1: it is all about the object

If you open `python` and type `1 + 2`, you will get `3`. If you open `R` and type `1 + 2`, you will get `3`. This is because `1` and `2` are objects and `+` is a method that is defined for these objects. Since `python` is more object-oriented than `R`, the syntax and structure of `python` is more consistent and predictable. In `R`, the syntax and structure is more flexible and less predictable. This gives `R` a lot of power and flexibility, but it also makes `R` more difficult to learn and use and sometimes very confusing for beginners.

Let's see some examples.

```bash
michael@Geothe-Uni-Ubuntu:~/Github/mysite$ python
Python 3.10.10 (main, Nov 13 2023, 21:20:19) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 1+2
3
```

The above code works because `1` and `2` are objects and `+` is a method that is defined for these objects. See the following code.

```python
>>> # assign an integer to an variable
>>> a = 2
>>> dir(a)
['__abs__', '__add__', '__and__', '__bool__', ... '__sizeof__', '__str__', ...]
>>> a.__str__()
'2'
>>> a.__add__(1)
3
>>> a.__neg__()
-2
>>> a.__sub__(1)
1
```

The special method `__str__` is called when you use `print(a)`. The special method `__add__` is called when you use `a + 1`. The special method `__sub__` is called when you use `a - 1`. The special method `__neg__` is called when you use `-a`. The special method `__bool__` is called when you use `if a:`. 

Since both `python` and `R` are both interpreted languages, they rely on the REPL (Read-Eval-Print Loop) to execute code with a virtual machine. The virtual machine is responsible for interpreting the code and executing it. The virtual machine is also responsible for managing the memory and the objects that are created and destroyed. It is also responsible for managing the methods that are defined for the objects. We will not go into the details of how the virtual machine works, but knowing this will lead us to appreciate the _front-end_ and _back-end_ of interpreting languages, such as `python` and `R`.

What is the _front-end_ and _back-end_ of interpreting languages? You can understand the _front-end_ and _back-end_ of interpreting languages by comparing them to the _front-end_ and _back-end_ of web development. The _front-end_ of web development is responsible for the user interface and the user experience. The _back-end_ of web development is responsible for the server and the database (or in programming language, the computing and the memory). 

In `python` and `R`, the language designers have to think about how to design the _font-end_ and _back-end_ of the language. We know both `python` and `R` used `C` as the back-end. The _front-end_ of `python` is more object-oriented and the _front-end_ of `R` is more functional.

Let's see the following code.

```bash
michael@Geothe-Uni-Ubuntu:~/Github/mysite$ R

R version 4.3.2 (2023-10-31) -- "Eye Holes"
Copyright (C) 2023 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.
> 1+2
[1] 3
> a <- 2
> a
[1] 2
> class(a)
[1] "numeric"
> attributes(a)
NULL
> typeof(a)
[1] "double"
> typeof(a + 3)
[1] "double"
> names(a)
NULL
> head(a)
[1] 2
> attr(a, 'names') <- 'hello'
> names(a)
[1] "hello"
> print(a)
hello 
    2 
> a
hello 
    2 
```

The above code shows that `R` is more functional and less object-oriented. The object `a` is a numeric object and it has no attributes. However, we can assign an attribute to `a` by using the `attr` function. The `attr` function is a special function that could be used to assign an attribute to an object. 

There is no such thing as `__add__` or `__str__` in `R`. Meanwhile, there is no such function as `attr` in `python`. This is because `R` is more functional and `python` is more object-oriented. In `Python`, you have to follow the object-oriented way to define a method for an object. In `R`, you can use the functional way to define a method for an object.

For people who have started with `C++` or `Java`, they may find `python` more familiar and easier to learn. For people who have started with `Lisp` or `Haskell`, they may find `R` more familiar and easier to learn. `Javascript` is quite special as it is also mixed with functional and object-oriented features but is more flexible and extensible than both `python` and `R`.


Now, let's see the following code.

```bash
>>> 'a' + 'b'
'ab'
>>> 'a'.__add__('b')
'ab'
```

However, in `R`, you cannot do the following.

```bash
> 'a' + 'b'
Error in "a" + "b" : non-numeric argument to binary operator
```

The reason is that `+` is not defined for `character` objects in `R`. However, you can use the `paste` function to concatenate two `character` objects.

```bash
> paste('a', 'b')
[1] "a b"
> paste0('a', 'b')
[1] "ab"
```

I think now you should very clear why we say `R` is more functional and `python` is more object-oriented. In `R`, you have to use the `paste` function to concatenate two `character` objects. In `python`, you can use the `+` operator to concatenate two `string` objects because `+` is defined for `string` objects based on the `__add__` method.

Because `Python` is more object-oriented, the object is more contained and the method is more predictable. By more contained, I mean that the object is more controlled by its own method and attributes it has. It will not be affected or contaminated that much by the environment or global methods (I am not an expert of language design, so I may not use the correct terms here). For instance, in `R` we have `+`, which is a binary operator that is defined for `numeric` objects at global level. However, in `Python`, we have `__add__`, which is a method that is defined for `int` objects or `float` objects or `str` objects.

Now, it is time to review the quote from John Chambers. John McKinley Chambers is the creator of the S programming language, and core member of the R programming language project. He was awarded the 1998 ACM Software System Award for developing S.

> Everything that exists is an object in R and everything that happens is a function call in R. -- John Chambers

We have elaborated the first part of the quote. Now, let's see the second part of the quote.


```bash
> `+`(3, 5)
[1] 8
> `%+%` <- function(a, b){
+ paste0(a, b)}
> 'a' %+% 'b'
[1] "ab"
```

If see the above code as a `python` programmer, you may find it very strange. I mean what is this `%+%` thing? According to the [official document](https://cran.r-project.org/doc/manuals/R-lang.html), this is a special way for defining your own binary operator. This kind of thing is not available in `python`. In `python`, you can define your own method for your own object, but you cannot define operator at global level (maybe you can, but I don't know how to do it).

All those things we covered here explains why you have this kind of pipeline in `R`.

```R
library(data.table)
library(magrittr)

iris %>% 
  as.data.table() %>% 
  .[, .(mean(Sepal.Length)), by = Species]
```

However, in `python`, your pipeline is more like this, which is just chaining the methods of the object.

```python
import pandas as pd

iris = pd.read_csv('.../iris.csv')
iris.groupby('Species')['Sepal.Length'].mean()
```


> Note in R, that some attributes (namely class, comment, dim, dimnames, names, row.names and tsp) are treated specially and have restrictions on the values which can be set. (Note that this is not true of levels which should be set for factors via the levels replacement function.)

## Advanced standpoint 2: programming = data structure + algorithm

_Algorithms + Data Structures = Programs_ is a 1976 book written by Niklaus Wirth covering some of the fundamental topics of system engineering, computer programming, particularly that algorithms and data structures are inherently related. 

Even for a simple task, algorithms (let's say a simple for loop) and data structures (let's say a simple vector) are always involved. Therefore, when you do programming or data analysis, you are always dealing with data structures and algorithms. This means we should be aware of the data sctures we are dealing with and then we could come up syntax and algorithm to solve the problem.

The whole community is now moving towards higher awareness of data structures and types. That's why we have `pydantic` in `python` and `typescript` in web development. Because `R` is a mix of functional and object-oriented, it is quite difficult to define a strict data structure in `R`. We will see many examples in the following sections. This makes `R` more flexbile but also more restrictive for certain tasks. Therefore, we will see `R` will alway dominate the data science and statistics field, but it will never be a good language for web development or system engineering.

> It should be this way as `R` was designed for data analysis and statistics, not for web development or system engineering. In a nutshell, `R` is not a general-purpose programming language like `python` or `javascript` but a domain-specific language like `SQL` or `LaTeX`.


Before I give you some examples, let's review the following table (taken from [this video](https://youtu.be/92YFZhKSJq0?si=tW6QvsYhO_cK_fTd){:target="_blank"}).

| Data structure | dimension | class | attributes/methods | mixed data types |
|----------------|-----------|-------|------------| -----------------|
| vector         | 1         | atomic| length     | no |
| matrix         | 2         | matrix| dim, dimnames | no |
| array          | n       | array | dim, dimnames | no |
| list           | n        | list  | length, names | yes |
| factor         | 1         | factor| levels, labels | no |
| table          | 2         | table | dim, dimnames, names ...| yes |
| data.frame     | 2         | data.frame | dim, names, rownames ... | yes |
| data.table     | 2         | data.table | dim, names, rownames ... | yes |


Knowing the data structures is extremely important as it will help you to manipulate data more efficiently and effectively.


## data.table from an advanced standpoint

Now, let's learn `data.table` from an advanced standpoint. The vignettes of `data.table` are very comprehensive and well-written. Please read them if you are new to `data.table`, here is the [link](https://cran.r-project.org/web/packages/data.table/vignettes/). In this post, I will try to summarize what I have learned about `data.table` into a framework from an advanced standpoint.

The starting point is same as the vignetts, you have to know the following syntax.

```R
DT[i, j, by]

##   R:                 i                 j        by
## SQL:  where | order by   select | update  group by
```

Notice that `[]` looks like a symbol of a rectangle, which already gives you a hint that `data.table` is all about data structure. We are working with a rectangle of data, which is a 2D data structure. This is the first thing you have to know about `data.table`. It is a 2D data structure. `data.table` is the enhanced version of `data.frame`. Therefore, many of the syntax and methods of `data.frame` are also available in `data.table`.

The framework I am introducing here is based on the following two principles:

1. visualize a matrix in your mind
2. think about the data structure either in a row-wise or column-wise manner

let's look at the following matrix

| id | name| ... |height| weight| ... | age | ... |
|----|-----|-----|------|-------|-----|-----|-----|
| 1  | John| ... | 180  | 70    | ... | 25  | ... |
|   | | ... |  |  65 | ... |  | ... |
|   | | ... |  |  75 | ... |  | ... |
|   | | ... |  |  80 | ... |  | ... |
| 78 | Mary| ... | 160  | 50    | ... | 30  | ... |

### Checking missing values for each column

Now, let's say you want to check the missing values for each column. You can do it in a row-wise manner or column-wise manner.


```R
# column-wise manner
dt %>%
    .[, lapply(.SD, function(x) sum(is.na(x))), .SDcols = names(dt)] %>%
    kable()

dt %>%
    is.na() %>% 
    colSums() %>% kable()

# row-wise manner
dt %>%
    .[, .(n_missing = sum(is.na(.SD))), by = id] %>% 
    kable()

dt %>%
    is.na() %>%
    rowSums() %>% kable()
```

I think column-wise manner is more intuitive and easier to understand. However, sometimes you might need to do operations in a row-wise manner. Since we could always transpose the matrix, we could always do operations in a column-wise manner.


### Checking the summary statistics for selected columns

Now, let's say you want to check the summary statistics for selected columns. The most common way is to use the `summary` function with `lapply` function. However, if you try the following code, you will get an error.

```R
dt %>%
    .[, lapply(.SD, summary), .SDcols = c('height', 'weight', 'age')] %>%
    kable()
# Error in dimnames(x) <- dnx: 'dimnames' applied to non-array
# Traceback:
```

However, the above code does not work because `summary` function returns a `table` for each column and the dimension of the `table` of each column is not the same. Therefore, you will get an error.

However, if you try the following code, you will still get an error.

```R
dt %>% 
    na.omit() %>%
    .[, lapply(.SD, summary), .SDcols = c('height', 'weight', 'age')] %>%
    kable()
# 'dimnames' applied to non-array
```

This is because `summary` function returns a `table` for each column and not a `list`. Therefore, you will get an error. If we run the following code, we get:

```R
dt$height %>% summary() %>% attributes()

# $names
# 'Min.''1st Qu.''Median''Mean''3rd Qu.''Max.''NA\'s'
# $class
# 'summaryDefault''table'
```

As you can see the class of the output of `summary` function is `table` and not `list`. Many statistical summary in `R` returns a `table` and not a `list`. Therefore, you have to be careful when you use `lapply` function with `summary` function.

> This shows that knowing the data structure is extremely important. If you know the output of a function is a `table` and not a `list`, you will not make the mistake of using `lapply` function with it.

The following code works.

```R
dt %>% 
    na.omit() %>%
    .[, sapply(.SD, summary), .SDcols = c('height', 'weight', 'age')] %>%
    kable()
```


|        |   height|    weight|  age|
|:-------|--------:|---------:|----:|
|Min.    | 132.3291|  39.58384| 18.0|
|1st Qu. | 163.8469|  63.31751| 30.0|
|Median  | 170.3699|  69.89749| 41.0|
|Mean    | 170.0986|  69.70877| 41.1|
|3rd Qu. | 176.9779|  76.40854| 54.0|
|Max.    | 198.5624| 105.43519| 64.0|

The above code works because:

1. `summary` function returns a `table` and not a `list`
2. `sapply` function is used to apply the `summary` function to each column of the `data.table` and the output is a `matrix` and not a `list`
3. since we put the output of `sapply` function into a `data.table`, the `matrix` is automatically converted into a `data.table` and the `dim` attribute is automatically added to the `data.table`
4. we use `na.omit` function to remove the missing values before we apply the `summary` function to the `data.table` as the dimension of the `table` of each column is not the same when there are missing values


> As long as j returns a list, each element of the list will become a column in the resulting data.table. If the result is a matrix, the columns of the matrix will become columns in the resulting data.table. 


### Order the count within each group

Now, we want to order the count for categorical variables such as `year`, `education` 
and `degree` within each group. 

Let's run the following code.

```R
dt %>%
    .[, year := year(dob)] %>%
    .[, .N, by = .(year, education, gender)] %>%
    .[order(year, -N)] %>%
    peep_head()
```


| year|education |gender |  N|
|:----|:---------:|:------:|--:|
| 1950|Bachelors |M      |  5|
| 1950|Bachelors |F      |  4|
| 1950|Masters   |M      |  3|
| 1950|PhD       |M      |  3|
| 1950|Masters   |F      |  3|

As you can see that we have ordered the year and the count. However, This is what we want. We want to order the count within each group, meaning, we should have rows that be ordered as a subgroup based on year, education and gender. Or what we want is something like this.

| year|education |gender |  N|
|:----|:---------:|:------:|--:|
| 1950|Bachelors |M      |  5|
| 1950|Bachelors |F      |  4|
| 1950|Masters   |M      |  3|
| 1950|Masters   |F      |  3|
| 1950|PhD       |M      |  3|
| 1950|PhD       |F      |  2|
| 1950|Masters   |F      |  3|

You might try the following code.

```R
dt %>%
    .[, year := year(dob)] %>%
    .[, .N, by = .(year, education, gender)] %>%
    .[order(year, gender, -N)] %>%
    peep_head()
```

It still does not work as `N` is always ordered at the global level. The following code works.

```R
dt %>%
    .[, year := year(dob)] %>%
    .[, .N, by = .(year, education gender)] %>%
    .[order(year, education)] %>%
    .[order(year, education, -N)] %>%
    peep_head()
```

| year|education   |gender |  N|
|:----|:-----------:|:------:|--:|
| 1950|Bachelors   |M      |  5|
| 1950|Bachelors   |F      |  4|
| 1950|High School |M      |  2|
| 1950|High School |F      |  2|
| 1950|Masters     |M      |  3|


But what if we want to order `N` at the global level but at the same time we want to order `gender` within each group? The following code works.

```R
dt %>%
    .[, year := year(dob)] %>%
    .[, .N, by = .(year, education, gender)] %>%
    .[, n_sum := sum(N), by = .(year, education)] %>%
    .[order(year, -n_sum, -N)] %>% peep_head(10)
```

The above code works because we have created a new column `n_sum` that is the sum of `N` within each group. Therefore, we can order `n_sum` at the global level and `N` within each group.

| year|education   |gender |  N| n_sum|
|:----|:-----------:|:------:|--:|-----:|
| 1950|Bachelors   |M      |  5|     9|
| 1950|Bachelors   |F      |  4|     9|
| 1950|Masters     |M      |  3|     6|
| 1950|Masters     |F      |  3|     6|
| 1950|High School |M      |  2|     4|
| 1950|High School |F      |  2|     4|
| 1950|PhD         |M      |  3|     3|
| 1951|Masters     |F      |  3|     5|
| 1951|Masters     |M      |  2|     5|
| 1951|Bachelors   |M      |  2|     3|
| 1951|PhD         |F      |  2|     3|
| 1951|PhD         |M      |  1|     3|

The above table tells us that for each year which education has the most number of people and which gender has the most number of people.

I used this example to introduce the following principle.

> Fundamental theorem of software engineering: we can solve any problem by introducing an extra level of indirection (David Wheeler)

In this example, we have introduced an extra level of indirection by creating a new column `n_sum` that is the sum of `N` within each group. This allows us to order `n_sum` at the global level and `N` within each group.


### Working with nested data

This is a very common task in data analysis, which is also known as unnesting or flattening. This kind of task is quite challenging for almost all programming languages.

Suppose you have one column that contains a nested json data, which looks like the following one:

```
# for one cell
{
  "name": "John",
  "dob": "1950-01-01",
  "education": [
    {
      "degree": "Bachelors",
      "year": 1970
    },
    {
      "degree": "Masters",
      "year": 1975
    }
  ]
  "working_experience": [
    {
      "company": "Google",
      "year": 1990
    },
    {
      "company": "Facebook",
      "year": 1995
    }
  ]
}

# another cell
{
    "name": "Mary",
    "dob": "1955-01-01",
    "education": [
        {
        "degree": "Bachelors",
        "year": 1975
        }
    ]
    "working_experience": [
        {
        "company": "Amazon",
        "year": 1995
        },
        {
        "company": "Microsoft",
        "year": 2000
        },
        {
        "company": "Apple",
        "year": 2005
        }
    ]
}
```

As you can see that the `education` and `working_experience` are nested data they also have different length. Based on my experience, there is no easy way in `R` to flatten the nested data. Since the nested json has `keys` and also `lists`, it is quite difficult to flatten the nested data in `R`.

However, if you use `python`, you can use the `pandas` library to flatten the nested data. The `json_normalize` function and `explode` function are very handy for this kind of task. In `duckdb`, you can also find the `flatten` function that is very handy for this kind of task.

> This again shows that data structure is extremely important. Because `R` does not have dictionary-like data structure, it is quite difficult to flatten the nested data in `R`. However, `python` has dictionary-like data structure, so it is quite easy to flatten the nested data in `python`.