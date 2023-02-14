---
title: Create a Virtual Environment
subtitle: Quick guide on how to create a virtual environment in Python
layout: blog_default
date: 2022-10-17
keywords: Python, Virtual Environment
published: true
tags: python venv
---

Here are the commands to create a virtual environment in Python: 


```bash
python3 -m venv myvenv
```

After than, you can do:

```bash
source myvenv/bin/activate
```


On Ubuntu, sometimes (very rarely) you need to do this:

```bash
python3 -m venv --without-pip myvenv
```

After than, you can do:

```bash
source myvenv/bin/activate
```

###  Install packages

please check `pip` path first:

```bash
which pip
```

If your pip path is within the virtual environment, then you can do 

```bash
pip install
```

If not, please run

```bash
python -m pip install
```