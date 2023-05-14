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


