---
title: Github Pages with Mkdocs 
subtitle: I used Mkdocs to build my webiste for a while before switching to Jekyll. 
layout: blog_default
date: 2022-10-19
keywords: blogging, writing, Mkdocs, Mkdocs-material
published: true
tags: mkdocs github-page website-tool
---


If you want to create a website with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/){:target="_blank"} and host it on Github Pages,
you could follow the instructions in this post: 

* [Setup](#setup)
* [Installation](#installation)
* [Create a website](#create-a-website)
* [Publish it on Github Pages](#publish-it-on-github-pages)
* [Customize Your website](#customize-your-website)

This post assumes that you know how to use `Git` and `Github`. 

## Setup 

Please make sure you have `Python` and `VS Code`installed. Now, we will setup
the environment and install the essential packages.

### Create An Virtual Environment

If you have multiple versions of python, you need set python path first

```bash
PATH="/Library/Frameworks/Python.framework/Versions/3.x/bin:${PATH}"   
export PATH
```
where `3.x = 3.7 or 3.8 or 3.9`, then you could install `virtualenv` with `pip3`:

```bash
python3 -m pip install --upgrade pip
pip3 install virtualenv
which virtualenv  # check the version of your virtualenv
which python3  # check the version of your python3
```
After installing `virtualenv`, you could navigate to your working directory and 
create your virtual environment.
```bash 
virtualenv -p /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 venv
```
It's better to call it `venv`, so you could configure your `.gitignore` file automatically. 
Once you created it, then you could activate it.
```bash 
source venv/bin/activate
```
You could also set the alias to your virtual environment. 
```bash 
alias blogenv="source venv/bin/activate"
```
Then every time you type `blogenv`, your virtual environment will be activated. 

### Install MkDocs Material 

```bash
pip install mkdocs-material
```

Now, we need to freeze our dependencies. 

```bash
pip freeze > requirements.txt
```

## Create a website

After setting up the environment and installing essential packages, we could
make our websites.

```bash
mkdocs new .
```
This gives the following structure:

```
.
├─ docs/
│  └─ index.md
└─ mkdocs.yml
```

You can check your website locally:

```
mkdocs serve
```

## Publish it on Github pages

In your folder cloned from `Github`, it should look like this:

```
.
├── LICENSE
├── README.md
├── docs
│   ├── blog
│   ├── index.md
│   ├── javascripts
│   └── stylesheets
├── mkdocs.yml
├── requirements.txt
└── venv
    ├── bin
    ├── lib
    └── pyvenv.cfg
```

Here is my `configuration ymal` file:

```yaml
site_name: Innovation Lab

theme:
  name: material
  font:
    text: Source Sans Pro
    code: Roboto Mono
  palette: 
    - scheme: default
      # primary: teal
      toggle:
        icon: material/toggle-switch 
        name: Switch to dark mode
    - scheme: slate 
      # primary: teal
      toggle:
        icon: material/toggle-switch-off-outline
        name: Switch to light mode
  features:
    - navigation.instant
    - navigation.tabs
    # - toc.follow
    - content.code.annotate
  markdown_extentions:  # do not move this outside
    - pymdownx.highlight:  # it is very senstive to indention 
        anchor_linenums: false


extra_css:
  - stylesheets/extra.css

extra_javascript:
  - 'javascripts/headroom.js'
  - 'javascripts/activate_headroom.js'


plugins:
  - search:
      lang: en

markdown_extensions:
  - abbr
  - admonition
  - attr_list
  - def_list
  - footnotes
  - meta
  - md_in_html
  - toc:
      permalink: true
      toc_depth: 2
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.betterem:
      smart_enable: all
  - pymdownx.caret
  - pymdownx.details
  - pymdownx.emoji:
      emoji_generator: !!python/name:materialx.emoji.to_svg
      emoji_index: !!python/name:materialx.emoji.twemoji
  - pymdownx.inlinehilite
  - pymdownx.keys
  - pymdownx.magiclink:
      repo_url_shorthand: true
      user: squidfunk
      repo: mkdocs-material
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.tilde

nav:
  - Home: index.md
  - Readings:
    - readings/index.md
  - Blog:
    - 2022:
      - blog/2022/github-pages-mkdocs.md
  - Life Style:
    - lifestyle/index.md
```

When you publish it on the `Github Pages`, you need to create a `CI-CD` file like
the following one:

```yaml
name: Publish website via GitHub Pages
on:
  push:
    branches:
      - master 
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Master
        uses: actions/checkout@v2
      - name : Set up python 3.9
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Deploy
        run : |
          git pull
          mkdocs gh-deploy
```

Please choose `gh-pages` branch to publish your website. 

## Customize Your Website

If you want to change the layout and design of MkDocs material, you could 
read the official [documentation](https://squidfunk.github.io/mkdocs-material/reference/){:target="_blank"}.
Here is my customized `css` file. 


```css
.md-grid {
    max-width: 1157px;
}

@media screen and (min-width: 960px) and (max-width: 1221px)  {
    /* .md-nav--secondary {
        display: none
    } */
    .md-tabs {
        display: block;
      }
    .md-typeset {
        margin-left: 170px;
    }
}

@media screen and (min-width: 800px) and (max-width: 960px)  {
    .md-typeset {
        margin-left: 117px;
        margin-right: 117px;
    }
}

/* @media only screen and (min-width : 768px) and (max-width : 1220px){
    .md-grid {
        max-width: 700px;
    }
} */

/* ul.navbar-nav:nth-child(1) > li:nth-child(1) {
    display: none;
} */


.md-typeset p {
    font-size: 14pt;
    line-height: 1.5;
    --md-typeset-a-color: #009999;
}

[data-md-color-scheme="slate"] {
    --md-hue: 170; /* [0, 360] */
    background-color:#2b2f2b;  /*#303030*/
    color:#e5e4e2;
  }

/* :root > * {
    --md-typeset-a-color: #009999;
} */

.animated {
    -webkit-animation-duration: .5s;
    -moz-animation-duration: .5s;
    -o-animation-duration: .5s;
    animation-duration: .5s;
    -webkit-animation-fill-mode: both;
    -moz-animation-fill-mode: both;
    -o-animation-fill-mode: both;
    animation-fill-mode: both;
    will-change: transform, opacity
}

.animated.slideDown {
    -webkit-animation-name: slideDown;
    -moz-animation-name: slideDown;
    -o-animation-name: slideDown;
    animation-name: slideDown
}

.animated.slideUp {
    -webkit-animation-name: slideUp;
    -moz-animation-name: slideUp;
    -o-animation-name: slideUp;
    animation-name: slideUp
}

@-webkit-keyframes slideUp {
    0% {
        -webkit-transform: translateY(0)
    }
    100% {
        -webkit-transform: translateY(-100%)
    }
}

@-moz-keyframes slideUp {
    0% {
        -moz-transform: translateY(0)
    }
    100% {
        -moz-transform: translateY(-100%)
    }
}

@-o-keyframes slideUp {
    0% {
        -o-transform: translateY(0)
    }
    100% {
        -o-transform: translateY(-100%)
    }
}

@keyframes slideUp {
    0% {
        transform: translateY(0)
    }
    100% {
        transform: translateY(-100%)
    }
}

.animated.slideUp {
    -webkit-animation-name: slideUp;
    -moz-animation-name: slideUp;
    -o-animation-name: slideUp;
    animation-name: slideUp
}

@-webkit-keyframes slideDown {
    0% {
        -webkit-transform: translateY(-100%)
    }
    100% {
        -webkit-transform: translateY(0)
    }
}

@-moz-keyframes slideDown {
    0% {
        -moz-transform: translateY(-100%)
    }
    100% {
        -moz-transform: translateY(0)
    }
}

@-o-keyframes slideDown {
    0% {
        -o-transform: translateY(-100%)
    }
    100% {
        -o-transform: translateY(0)
    }
}

@keyframes slideDown {
    0% {
        transform: translateY(-100%)
    }
    100% {
        transform: translateY(0)
    }
}

.animated.slideDown {
    -webkit-animation-name: slideDown;
    -moz-animation-name: slideDown;
    -o-animation-name: slideDown;
    animation-name: slideDown
}
```

I also used `JavaScript` to hide the nav bar when you scroll down. You could do
it too by following this [Link](https://gist.github.com/oceanumeric/023f85b8040f2e8b6503da1dd30f5d9d){:target="_blank"}.

Here is the `css variables` that allow you to change color for different components.

```scss
////
/// Copyright (c) 2016-2022 Martin Donath <martin.donath@squidfunk.com>
///
/// Permission is hereby granted, free of charge, to any person obtaining a
/// copy of this software and associated documentation files (the "Software"),
/// to deal in the Software without restriction, including without limitation
/// the rights to use, copy, modify, merge, publish, distribute, sublicense,
/// and/or sell copies of the Software, and to permit persons to whom the
/// Software is furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
/// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS
////

// ----------------------------------------------------------------------------
// Rules
// ----------------------------------------------------------------------------

// Color variables
:root {
  @extend %root;
}

// ----------------------------------------------------------------------------

// Allow to explicitly use color schemes in nested content
[data-md-color-scheme="default"] {
  @extend %root;
}

// ----------------------------------------------------------------------------
// Placeholders
// ----------------------------------------------------------------------------

// Default theme, i.e. light mode
%root {

  // Default color shades
  --md-default-fg-color:               hsla(0, 0%, 0%, 0.87);
  --md-default-fg-color--light:        hsla(0, 0%, 0%, 0.54);
  --md-default-fg-color--lighter:      hsla(0, 0%, 0%, 0.32);
  --md-default-fg-color--lightest:     hsla(0, 0%, 0%, 0.07);
  --md-default-bg-color:               hsla(0, 0%, 100%, 1);
  --md-default-bg-color--light:        hsla(0, 0%, 100%, 0.7);
  --md-default-bg-color--lighter:      hsla(0, 0%, 100%, 0.3);
  --md-default-bg-color--lightest:     hsla(0, 0%, 100%, 0.12);

  // Primary color shades
  --md-primary-fg-color:               hsla(#{hex2hsl($clr-indigo-500)}, 1);
  --md-primary-fg-color--light:        hsla(#{hex2hsl($clr-indigo-400)}, 1);
  --md-primary-fg-color--dark:         hsla(#{hex2hsl($clr-indigo-700)}, 1);
  --md-primary-bg-color:               hsla(0, 0%, 100%, 1);
  --md-primary-bg-color--light:        hsla(0, 0%, 100%, 0.7);

  // Accent color shades
  --md-accent-fg-color:                hsla(#{hex2hsl($clr-indigo-a200)}, 1);
  --md-accent-fg-color--transparent:   hsla(#{hex2hsl($clr-indigo-a200)}, 0.1);
  --md-accent-bg-color:                hsla(0, 0%, 100%, 1);
  --md-accent-bg-color--light:         hsla(0, 0%, 100%, 0.7);

  // Code color shades
  --md-code-fg-color:                  hsla(200, 18%, 26%, 1);
  --md-code-bg-color:                  hsla(0, 0%, 96%, 1);

  // Code highlighting color shades
  --md-code-hl-color:                  hsla(#{hex2hsl($clr-yellow-a200)}, 0.5);
  --md-code-hl-number-color:           hsla(0, 67%, 50%, 1);
  --md-code-hl-special-color:          hsla(340, 83%, 47%, 1);
  --md-code-hl-function-color:         hsla(291, 45%, 50%, 1);
  --md-code-hl-constant-color:         hsla(250, 63%, 60%, 1);
  --md-code-hl-keyword-color:          hsla(219, 54%, 51%, 1);
  --md-code-hl-string-color:           hsla(150, 63%, 30%, 1);
  --md-code-hl-name-color:             var(--md-code-fg-color);
  --md-code-hl-operator-color:         var(--md-default-fg-color--light);
  --md-code-hl-punctuation-color:      var(--md-default-fg-color--light);
  --md-code-hl-comment-color:          var(--md-default-fg-color--light);
  --md-code-hl-generic-color:          var(--md-default-fg-color--light);
  --md-code-hl-variable-color:         var(--md-default-fg-color--light);

  // Typeset color shades
  --md-typeset-color:                  var(--md-default-fg-color);

  // Typeset `a` color shades
  --md-typeset-a-color:                var(--md-primary-fg-color);

  // Typeset `mark` color shades
  --md-typeset-mark-color:             hsla(#{hex2hsl($clr-yellow-a200)}, 0.5);

  // Typeset `del` and `ins` color shades
  --md-typeset-del-color:              hsla(6, 90%, 60%, 0.15);
  --md-typeset-ins-color:              hsla(150, 90%, 44%, 0.15);

  // Typeset `kbd` color shades
  --md-typeset-kbd-color:              hsla(0, 0%, 98%, 1);
  --md-typeset-kbd-accent-color:       hsla(0, 100%, 100%, 1);
  --md-typeset-kbd-border-color:       hsla(0, 0%, 72%, 1);

  // Typeset `table` color shades
  --md-typeset-table-color:            hsla(0, 0%, 0%, 0.12);

  // Admonition color shades
  --md-admonition-fg-color:            var(--md-default-fg-color);
  --md-admonition-bg-color:            var(--md-default-bg-color);

  // Footer color shades
  --md-footer-fg-color:                hsla(0, 0%, 100%, 1);
  --md-footer-fg-color--light:         hsla(0, 0%, 100%, 0.7);
  --md-footer-fg-color--lighter:       hsla(0, 0%, 100%, 0.3);
  --md-footer-bg-color:                hsla(0, 0%, 0%, 0.87);
  --md-footer-bg-color--dark:          hsla(0, 0%, 0%, 0.32);

  // Shadow depth 1
  --md-shadow-z1:
    0 #{px2rem(4px)} #{px2rem(10px)} hsla(0, 0%, 0%, 0.05),
    0 0              #{px2rem(1px)}  hsla(0, 0%, 0%, 0.1);

  // Shadow depth 2
  --md-shadow-z2:
    0 #{px2rem(4px)} #{px2rem(10px)} hsla(0, 0%, 0%, 0.1),
    0 0              #{px2rem(1px)}  hsla(0, 0%, 0%, 0.25);

  // Shadow depth 3
  --md-shadow-z3:
    0 #{px2rem(4px)} #{px2rem(10px)} hsla(0, 0%, 0%, 0.2),
    0 0              #{px2rem(1px)}  hsla(0, 0%, 0%, 0.35);
}
```

### Render Math Equations

{% katexmm %}

A random variable $X$ is continuous if its distribution function $F(x) = P(X \leq x)$ 
can be written as

$$
F(x) = \int_{-\infty}^x f(u) d u
$$

for some integrable $f: R \to [0, \infty)$. The function $f$ is 
called the _probability density function_ of the continuous random variable $X$.
For instance, the probability density function of the normal distribution is

$$f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{- \frac{(x-\mu)^2}{2 \sigma^2}}$$

For the following multivariate linear regression model,

$$
\begin{aligned}
    \begin{bmatrix}
        y_1 \\
        y_2 \\
        \vdots \\
        y_t \\
        \vdots  \\ 
        y_N
    \end{bmatrix} = \begin{bmatrix}
        1 & x_{12} & \cdots & x_{1k} \\
        1 & x_{22} & \cdots & x_{2k} \\
        \vdots & \vdots & \vdots \\
        1 & x_{t2} & \cdots & x_{tk} \\ 
        \vdots & \vdots & \vdots \\
        1 & x_{N2} & \cdots & x_{Nk} 
    \end{bmatrix} \begin{bmatrix}
        \beta_1 \\ 
        \beta_2 \\
        \vdots \\
        \beta_t \\ 
        \vdots \\
        \beta_k
    \end{bmatrix} + \begin{bmatrix}
        u_1 \\
        u_2 \\
        \vdots \\
        u_t \\
        \vdots  \\ 
        u_N
    \end{bmatrix}
\end{aligned} 
$$ 

or simply:

$$
\begin{aligned}
    Y = X \beta + u 
\end{aligned}
$$

where, $Y$ is a $N \times 1$ vector, $X$ is a $N \times k$ matrix, and 
$\beta$ is a $k \times 1$ vector. 

{% endkatexmm %}