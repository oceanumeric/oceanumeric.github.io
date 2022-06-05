---
title: Using Jupyter Notebook in MkDocs Material
---

Would it great if we could use Jupyter Notebook in our website? In this post, 
I will share what we could enrich a website built by MkDocs Material 
with Jupyter Notebook step by step:

* [install the package and configure the website](#install-the-package-and-update-configuration-file)
* [use Material design in Jupyter Notebook](#using-material-theme-style-in-jupyter-notebook)
* [customize your Jupyter Notebook with extrac `css` file](#customize-your-jupyter-notebook) 

## Install the package and update configuration file 

First, you need to install the plugin.
```bash
pip install mkdocs-jupyter
```
Then update your `mkdocs.yml` as by following this example. 
```yaml
nav:
- Home: index.md
- Notebook page: notebook.ipynb  
- Mathematics:
    - math/index.md
    - probability:
      - math/prob/history-probability.md
      - math/prob/generating-function.ipynb

plugins:
  - mkdocs-jupyter
```

## Using Material theme style in Jupyter Notebook 

In the Jupyter Notebook, the syntax rule specified by Material for MkDocs
does not work as we are not editing `markdown` file anymore. If you want
to use Material design, you need to call its `css class` directly. For instance,
to have an admonition, you could use the following code. 

!!! note

    ```html
    <div class="admonition note">
        <p class="admonition-title">Note</p>
        <p>
            If two distributions are similar, then their entropies are 
            similar, implies the KL divergence with respect to two 
            distributions will be smaller. And vica versa. In Variational
            Inference, the whole idea is to <strong>minimize</strong> KL 
            divergence so that our approximating distribution $q(\theta)$
            can be made similar to $p(\theta|D)$.
        </p>
    </div>
    ```

To have a collapsible admonition in a web page built with Jupyter Notebook,
you can use the following code. 
??? tip

    ```html
    <div class="result">
        <details class="quote"> 
            <summary>Quote</summary> 
            <p>
            A generating function is a device somewhat similar to a bag.
            Instead of carrying many little objects detachedly, which could
            be embarrassing, we put them all in a bag, and then we have only
            one object to carry, the bag.            
            </p>
            <p style="text-align:right">
            -- George Pólya
            </p>
        </details> 
    </div>
    ```

## Customize your Jupyter Notebook

Suppose you want to customize your jupyter notebook with your personal style,
such as `LaTex` style. You can just edit your `css` file and add a header into
the first `markdown cell` of your jupyter notebook. Here is an example I used
for my jupyter notebook. 

```html
<head>
<style>
    [data-md-color-scheme="default"] {
        /* only change colors when it is in light mode */
        background-color:#fffff8;
        .md-content > .md-typeset {
            color: #16291c;
            }
    }
    .md-content {
        /*support extra fonts */
        /*please download it first */
        font-family: 'Latin Modern Roman', Palatino, Times, serif;
    }
    .md-content > .md-typeset {
        font-size: 13pt;
        font-style: normal;
        font-weight: normal;
        text-align:justify;
    }
</style>
</head>
```

Here is a [demonstration](jupyter-demo.ipynb) that was built with Jupyter Notebook
with `Latex` Style. 

### Latex Style 

Google fonts does not have `Latin Modern Roman`, therefore you need to download
font by yourself and load it to `css` file. Here is my file tree:
```
├── blog
│   ├── 2022
│   │   ├── github-pages-mkdocs.md
│   │   └── jupyter-notebook-mkdocs.md
│   └── index.md
├── index.md
├── math
│   ├── index.md
│   └── prob
│       ├── generating-function.ipynb
│       └── history-probability.md
└── stylesheets
    ├── extra.css
    └── fonts
        ├── LM-bold-italic.ttf
        ├── LM-bold-italic.woff
        ├── LM-bold-italic.woff2
        ├── LM-bold.ttf
        ├── LM-bold.woff
        ├── LM-bold.woff2
        ├── LM-italic.ttf
        ├── LM-italic.woff
        ├── LM-italic.woff2
        ├── LM-regular.ttf
        ├── LM-regular.woff
        └── LM-regular.woff2
```

In your `extra.css` file, you need load fonts. 
```css
/* modern font */
@font-face {
    font-family: 'Latin Modern';
    font-style: normal;
    font-weight: normal;
    font-display: swap;
    src: url('./fonts/LM-regular.woff2') format('woff2'),
        url('./fonts/LM-regular.woff') format('woff'),
        url('./fonts/LM-regular.ttf') format('truetype');
}

@font-face {
    font-family: 'Latin Modern';
    font-style: italic;
    font-weight: normal;
    font-display: swap;
    src: url('./fonts/LM-italic.woff2') format('woff2'),
        url('./fonts/LM-italic.woff') format('woff'),
        url('./fonts/LM-italic.ttf') format('truetype');
}

@font-face {
    font-family: 'Latin Modern';
    font-style: normal;
    font-weight: bold;
    font-display: swap;
    src: url('./fonts/LM-bold.woff2') format('woff2'),
        url('./fonts/LM-bold.woff') format('woff'),
        url('./fonts/LM-bold.ttf') format('truetype');
}

@font-face {
    font-family: 'Latin Modern';
    font-style: italic;
    font-weight: bold;
    font-display: swap;
    src: url('./fonts/LM-bold-italic.woff2') format('woff2'),
        url('./fonts/LM-bold-italic.woff') format('woff'),
        url('./fonts/LM-bold-italic.ttf') format('truetype');
}
```

__References__:
<br>
[1] [Daniel Rodriguez](https://github.com/danielfrg/mkdocs-jupyter)
<br>
[2] [Stefan Gössner](https://goessner.github.io/mdmath/publication.html)