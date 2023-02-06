---
title: "Scientific Blogs: Jekyll, Mkdocs, R-Markdown or Quarto"
subtitle: What are the best tools to write and publish scientific or research blogs? I end up using Jekyll with Katex rendering on server-side. 
layout: blog_default
date: 2023-02-05
keywords: blogging, writing, Jekyll, Mkdocs, Quarto, Jupyter, Notebook, R Markdown, Python, R, Hexo, Hugo 
tags: post tool github-page jekyll
published: true
---

If you want to create a website to organize your writings, technological
notes, research ideas, etc., you might struggle over choosing different 
frameworks to do a 'proper' job. Here are some well-known frameworks 
in this business: Hexo, Hugo, Jekyll, Mkdocs, Quarto, R Markdown, Jupyter Book
You name it! 

There are many posts talking about pros and cons for each framework, here
are some of them:

- [We don't talk about Quarto](https://www.apreshill.com/blog/2022-04-we-dont-talk-about-quarto/){:target="_blank"}
- [Jekyll vs Hugo vs Mkdocs](http://make.waaglabs.nl/fablab/interns/2019/michelle-vossen/other/2019/10/24/comparison-of-static-website-generators.html){:target="_blank"}
- [Porting a distill blog to quarto](https://blog.djnavarro.net/posts/2022-04-20_porting-to-quarto/){:target="_blank"}
- [Enable Blogging Capabilities with Material for MkDocs](https://www.dirigible.io/blogs/2021/11/2/material-blogging-capabilities/#overriding-the-content-block){:target="_blank"}

Over the last five years, I have tried Hugo with academic theme, Mkdocs with 
material theme, R Bookdown, etc. In the end, I chose Jekyll over Mkdocs with 
material theme. Why?

First, the performance score of Mkdocs with material theme is not very good.
Even though it does have fantastic plugins like `mkdocs-jupyter`, it makes
the website and each post become heavy. The following figure gives
the lighthouse performance score of landing page of my old website built
with Mkdocs Material. 

<div class='figure'>
    <img src="/images/blog/mkdocs1.png"
         alt="Screenshot of performance score"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Performance Score for
        Mkdocs Material
    </div>
</div>

Things get worse when I tried to write a mathematical blogs. Figure 2 speaks 
for itself. 

<div class='figure'>
    <img src="/images/blog/mkdocs2.png"
         alt="Screenshot of performance score for a math post"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Performance Score for
        a Mathematical Post Made by Mkdocs Material
    </div>
</div>

## Rendering math fast 

{% katexmm %}

Since many of my posts include lots of math equations. I have to make sure
the page will be rendered smoothly. With `Jupyter Notebook`, only `MathJax 2.7`
is supported, which makes it very slow when your post grows bigger. For example,
no one could finish reading [this post](https://nbviewer.org/github/oceanumeric/Applied-Mathematics/blob/main/volume1/Linear-Systems-and-Floating-Point-Arithmetic.ipynb){:target="_blank"} without
being distracted. 

I have tried to build up my own framework by leveraging `nbconvert`, which
does have some functions to make the post become clean, such as hiding
`In[], Out[]` in the post, providing `copy button` for the code block. However,
keeping the style consistent and pre-rendering math equation become very
difficult and time consuming. I did manage to render math equations on the
server side by using `mathjax-node-page`(here is the [repo](https://github.com/oceanumeric/nbtransfer){:target="_blank"}
if you want to do the same thing.). It turned out this issue is more
general than I thought {% cite fedorin joashc%}. 

Regards rendering math online, there are many options. Readers are recommended
checking those [demos](https://bourne2learn.com/math/){:target="_blank"}. In
the end, I was nudged to chose Jekyll because of the plugin 
[jekyll-katex](https://github.com/linjer/jekyll-katex){:target="_blank"} 
allows me to render math on server-side easily. The theme of this blog was
taken from Gundersen's blog {% cite Gundersen %}.  

Here are some examples of rendering math equations in the post. 

__Variance of the sum of multiple random variables.__ For random variables $X_i$
we have:

$$\text{Var}\left(\sum_{i=1}^{n}X_{i}\right)=\sum_{i=1}^{n}\sum_{j=1}^{n}\text{Cov}\left(X_{i},X_{j}\right)$$

Using the decomposition formula of variance, we have

$${\rm var} \left( \sum_{i=1}^{n} X_i \right) = E \left( \left[ \sum_{i=1}^{n} X_i \right]^2 \right) - \left[ E\left( \sum_{i=1}^{n} X_i \right) \right]^2$$

Now note that $(\sum_{i=1}^{n} a_i)^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} a_i a_j$, 
which is clear if you think about what you're doing when you calculate 

$$(a_1+...+a_n) \cdot (a_1+...+a_n),$$

by hand. Therefore,

$$E \left( \left[ \sum_{i=1}^{n} X_i \right]^2 \right) = E \left( \sum_{i=1}^{n} \sum_{j=1}^{n} X_i X_j \right) = \sum_{i=1}^{n} \sum_{j=1}^{n} E(X_i X_j)$$

similarly,

$$\left[ E\left( \sum_{i=1}^{n} X_i \right) \right]^2 = \left[ \sum_{i=1}^{n} E(X_i) \right]^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} E(X_i) E(X_j)
$$

So,

$${\rm var} \left( \sum_{i=1}^{n} X_i \right) = \sum_{i=1}^{n} \sum_{j=1}^{n} \big( E(X_i X_j)-E(X_i) E(X_j) \big) = \sum_{i=1}^{n} \sum_{j=1}^{n} {\rm cov}(X_i, X_j)$$

__Pairwise independence.__ In general:

$$\text{Var}\left(\sum_{i=1}^{n}X_{i}\right)=\sum_{i=1}^{n}\sum_{j=1}^{n}\text{Cov}\left(X_{i},X_{j}\right)$$

If $X_i$ and $X_j$ are independent then $\mathbb{E}X_{i}X_{j}=\mathbb{E}X_{i}\mathbb{E}X_{j}$
and consequently

$$\text{Cov}\left(X_{i},X_{j}\right):=\mathbb{E}X_{i}X_{j}-\mathbb{E}X_{i}\mathbb{E}X_{j}=0
$$

This leads to:

$$\text{Var}\left(\sum_{i=1}^{n}X_{i}\right)=\sum_{i=1}^{n}\text{Var}X_{i}
$$



## Set up the website

Unlike python's community, the tutorial documents for setting up
a website with `Github Pages` are not very easy to follow. It took me
a while to set up this website. Since all static site generators
are to organize files and templates, one has to understand how `Jekyll`
organize your markdown files and templates of themes. Now, let's generate
a static website step by step. 

- Step 1: Check ruby version `ruby -v` and make sure ruby installed

- Step 2: Install Jekyll `gem install bundler jekyll`; if you see errors like 
followings:

```
ModuleNotFoundError: No module named 'apt_pkg'
ModuleNotFoundError: No module named 'apt_inst'
```
This means you need link your packages:


```bash 
# go to the package folder
cd /usr/lib/python3/dist-packages
# check apt_pkg version
ls -l | grep apt_pkg
# check apt_inst version
ls -l | grep apt_inst
# link the package for your version
sudo cp apt_pkg.cpython-38-x86_64-linux-gnu.so apt_pkg.so
sudo cp apt_inst.cpython-38-x86_64-linux-gnu.so apt_inst.so
```

- Step 3: Create a repository called `<your-github-username>.github.io` 

- Step 4: Clone your repository and create a folder called `docs` within
this repository

```bash
cd github-repository
# you can call it something else, but docs is a good folder name 
mkdir docs 
cd docs 
# create the jekyll website
jekyll new . 
```

- Step 5: Understand the structure of your website

Generally, you have two kinds of folders:

1. folders with underscore like `_layouts`
2. folders without underscore like `images` and `assets` 

```bash
.
├── index.markdown  # home page - index 
├── _config.yml  # manager the configure options 
├── Gemfile  # manager the ruby environment 
├── Gemfile.lock  # lock the environment 
├── 404.html
├── about.markdown  # another landing page if you need it
├── assets  # js file 
│   └── katex.min.js
├── css  # css files 
│   ├── blog.css
│   ├── main.css
│   ├── markdown.css
│   └── trac.css
├── images  # images for your website 
│   ├── blog
│   │   ├── mkdocs1.png
│   │   └── mkdocs2.png
│   ├── favicon.png
│   └── inclusion-exclusion.png
├── blog  # blog section 
│   └── index.html
├── _blogs  # all blogs 
│   ├── example.md
│   └── jekyll-mkdocs-quarto.md
├── _bibliography
│   └── references.bib
├── _includes  # header and nav templates 
│   ├── header.html
│   └── nav.html
├── _layouts  # layout template for different sections 
│   ├── blog_default.html
│   └── default.html
└── _site  # built site contents goes into this folder
```

Let's say you want to build up a section or portfolio for your
website called `teaching` section. Then you need:

1. `teaching` folder to set up the landing page
2. `_teaching` folder to store different contents
3. a html file in `_layout` folder if you want a different layout 

To learn how this website was built, please visit my [repo](https://github.com/oceanumeric/oceanumeric.github.io){:target="_blank"}. 

- Step 6: Install plugins and configure them in `_config.yml` file. 

- Step 7: Set up CICD (Github Actions)

Here is my CICD file in `.github/workflows`: 

```yml
name: Build and Deploy My Jekyll Site to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  jekyll:
    runs-on: ubuntu-latest

    permissions:
      contents: write

    steps:
      - name: 📂 setup github actions 
        uses: actions/checkout@v2
      
      # setup ruby environment 
      - name: 💎 setup ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: 3.1.2

      - name: 🔨 install dependencies & build site
        uses: limjh16/jekyll-action-ts@v2
        with:
          enable_cache: true

      - name: 🚀 deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./_site
```



{% endkatexmm %}
