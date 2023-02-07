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

- [An unpleasant experience](#an-unpleasant-experience)
- [The ideal environment](#the-ideal-environment)
- [Build a docker image](#build-a-docker-image)
- [Using datascience-notebook image](#using-datascience-notebook-image)

## An unpleasant experience 

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

## The ideal environment 

The idea working environment for data scientists should be of accessible,
reusable, having no trouble of installing most of packages, and could
be updated for new projects. To have this idea working environment, we could
buy a laptop or server and install everything we need and do not install
other softwares that are not needed for data analysis. 

However, I do not have money for this. Even I had money to buy all those
hardwares, it is time consuming to switch between them. Therefore, we need
Docker. 


## Build a docker image

With Docker installed, we can create a `Dockerfile` to create an image.
For instance, the following one is taken from [Jupyter development team](https://github.com/jupyter/docker-stacks/blob/main/docker-stacks-foundation/Dockerfile){:target="_blank"}. I think knowing 
how to build a docker image is quite essential for being a data scientist or data
engineer. We will walk through the following Dockerfile and explain what 
it does. 


```docker
# This is just an example, do not run it! 
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

################ Setup a base operating system ###########
# Ubuntu 22.04 (jammy)
# https://hub.docker.com/_/ubuntu/tags?page=1&name=jammy
# choose ubuntu:22.04 as base operating system 
ARG ROOT_CONTAINER=ubuntu:22.04

FROM $ROOT_CONTAINER

# add docker user information
# ARG - define arguments build-time variables. 
# you can set up user id otherwise everytime you run docker you will have
# different ids, such as root@71b22c5215f2; root@79a18ea42f06; 
LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# you will see this user when you run docker container 
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# running shell 
# This setting prevents errors in a pipeline from being masked.
# image you apt-get install, but error was not returned 
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# set up the user for Ubuntu
# meaning we will install all softwares in our Ubuntu image as root user 
USER root

# envrironment variable for installation 
# we do not need to interact during installing process such as answering [y/n]
ENV DEBIAN_FRONTEND noninteractive

################### Install Softwares in Ubutun #########################
# if you have ever used Ubuntun
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - bzip2 is necessary to extract the micromamba executable.
    bzip2 \
    p7zip-full \
    ca-certificates \
    locales \
    sudo \
    # - tini is installed as a helpful container entrypoint that reaps zombie
    #   processes and such of the actual executable we want to start, see
    #   https://github.com/krallin/tini#why-tini for details.
    tini \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen


# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH="${CONDA_DIR}/bin:${PATH}" \
    HOME="/home/${NB_USER}"

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
# .........
# .........
# more comannd
```

The original `Dockerfile` is very long, we will not discuss everything here.
Once the operating system and dependencies become complex, we could reply
on images built by others. Jupyter development team has built several 
images, such as [datascience-notebook](https://hub.docker.com/r/jupyter/datascience-notebook){:target="_blank"},
[tensorflow-notebook](https://hub.docker.com/r/jupyter/tensorflow-notebook){:target="_blank"}, 
and [r-notebook](https://hub.docker.com/r/jupyter/r-notebook){:target="_blank"}. 
With those images, you can customize or build new images based on them. 


## Using datascience-notebook image

Since `datascience-notebook` image has been downloaded (pulled) 10M+, I will
just use this image instead of building my own programming environment. When 
you are using this image, you have two ways to run it:

- use it as an isolated container and save all files within this container
- use it as a computation station and save and access files locally

Before we run everything, make sure you pulled the image by running
`docker pull jupyter/datascience-notebook`.


#### Using it without accessing local files (an isolated container)

You could skip this section as most users need to access the local files
from their own computer (or whatever working station) and interact with
the container. The following command will not allow you read a local file
and analyze your data. You have to download everything within it
after running the container (see you clone a git repository within it
and work on it). 

For the scenario one, we could run 

```bash
# -it means iteractive with terminal
# --rm means removing container after exit
# start.sh set up the starting point
# default one is jupyter lab 
docker run -it --rm jupyter/datascience-notebook start.sh bash
```

Once you run it, you should get

```
Entered start.sh with args: bash
Executing the command: bash
(base) jovyan@72e83ed33c2f:~$ 
```

With the username 'jovyang' (username set in the Dockerfile) and id as
`72e83ed33c2f` and dollar sign $, now you are having a `Ubuntu` system
with all essential packages installed. 

As the following code shows, there are no extra files in our container. 

```bash
(base) jovyan@72e83ed33c2f:~$ ls
work
(base) jovyan@72e83ed33c2f:~$ cd work
(base) jovyan@72e83ed33c2f:~/work$ ls
(base) jovyan@72e83ed33c2f:~/work$ 
```

Sometime you need to open multiple terminal windows, you can do this:

- step 1: run a container

```bash
# the start.sh bash is option only for jupyter/datascience-notebook image
docker run -it --rm --name notebook2 jupyter/datascience-notebook start.sh bash
```

- step 2: open another terminal and check the running containers

```bash
docker ps
```

It will show I have a container called `notebook2` running and could be interacted.
Then run the following command gives us another terminal:

```bash
docker exec -it notebook2 /bin/bash
(base) jovyan@20a1058c5415:~$ 
# turn it off
exit 
```
Notice the user ID is the same one as we are interacting with the same container. 


#### Using it as a computation station and save and access files locally

Most time, we want to use this container as a computation station and
save and access local files with no restriction. Then we need to run

```bash
# you can call whatever youu want --name <your name>
# -v meaning we now links current folder (${PWD}) with jovyan/work
docker run -it --rm --name notebook2 -v "${PWD}":/home/jovyan/work jupyter/datascience-notebook start.sh bash

Entered start.sh with args: bash
Executing the command: bash
(base) jovyan@2623ed260b84:~$ ls
work
(base) jovyan@2623ed260b84:~$ cd work
(base) jovyan@2623ed260b84:~/work$ ls
blogenv  Dockerfile  docs  LICENSE  notebooks  README.md
(base) jovyan@2623ed260b84:~/work$ 
``` 
You can see my local files are shown within the container. 





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