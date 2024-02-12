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