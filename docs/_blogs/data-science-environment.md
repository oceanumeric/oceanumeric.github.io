---
title: Programming Environment for Data Science
subtitle: What should we do when we realize some packages from Python or R could not be installed in our local machine?
layout: blog_default
date: 2023-01-07
keywords: data science, Python, R, packages, dependencies, Docker, container, ssh, remote
published: true
tags: r python data-science docker
---

Every time you start a new project or analyze a new dataset, you might come
to a situation that you need to `install` a new package. However, sometimes
installing a package could take you a full day (I believe every programmer
has this kind of experience). To do data analysis with open source ecosystem,
we rely on so many packages. If anything in this long chain of dependence
goes wrong, we have to halt our project. 

Personally, I have to set up different kinds of programming environment
for different projects. For example, you might need a programming environment
with GPUs, you might need a programming environment that allows you to do
data analysis with `Python`, `R` and `Stata` at the same time. 

How could we make sure our operating system will not be messed up each time
when we install and uninstall different softwares. For instance, my current
project needs me to use `Python` and `R` together, and it annoyed me when
I failed to install a package in `R` called `devtools`. It turned out three
packages that `devtools` depends on could not be installed:

```r
Warning messages:
1: In install.packages("devtools") :
  installation of package 'ragg' had non-zero exit status
2: In install.packages("devtools") :
  installation of package 'pkgdown' had non-zero exit status
3: In install.packages("devtools") :
  installation of package 'devtools' had non-zero exit status
```

I tried to fix this by google and still failed to install them after TWO hours.
Then, I decided to use [Docker](https://www.docker.com/){:target="_blank"}. 
Docker takes away repetitive, mundane configuration tasks and is used throughout 
the development lifecycle for fast, easy and portable application 
development – desktop and cloud. 

If you knew `virtual envrionment` in `Python`, then Docker is the tool for
the operating system. With docker, you can have Linux image in your Windows
or MacOS system. Every time you run an image, you are creating a container
that could be:

- isolated operating system for a certain task
- managed and updated in terms of software and packages dependences 
- used locally or remotely as a server

## The ideal 

```
docker run -p 8888:8888 --name notebook -v "${PWD}":/home/jovyan/work -e JUPYTER_ENABLE_LAB=yes  -it jupyter/datascience-notebook
```

```
docker run -it --rm -p 8888:8888 --name notebook -v "${PWD}":/home/jovyan/work jupyter/datascience-notebook
```

```
docker run -d --rm -p 8888:8888 -e JUPYTER_TOKEN=letmein --name notebook -v "${PWD}":/home/jovyan/work jupyter/datascience-notebook
```

```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --shm-size=8gb -p 7070:7070
```

```
docker run --ipc=host  -it --rm  -p 8888:8888
```

```
jupyter notebook --ip 0.0.0.0 --port=8888 --no-browser --allow-root
```