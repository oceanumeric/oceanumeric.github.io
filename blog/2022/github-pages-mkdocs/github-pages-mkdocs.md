---
title: Github Pages with MkDocs
---



If you want to create a website with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and host it on Github Pages,
you could follow the instructions in this post: 

* Setup
* Installation 
* Create a website
* Publish it on Github Pages
* Customize Your website

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
read the official [documentation](https://squidfunk.github.io/mkdocs-material/reference/).
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
it too by following this [Link](https://gist.github.com/oceanumeric/023f85b8040f2e8b6503da1dd30f5d9d).