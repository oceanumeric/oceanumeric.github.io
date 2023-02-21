---
title: Using R and Python in VS Code Like a Data Scientist
subtitle: Embracing the python and R community together needs smooth transference of our mentalities and skills from both community; I believe one could get best of both worlds in VS Code.
layout: blog_default
date: 2023-02-20
keywords: data science, python, R, VS code, stack tools, IDE
published: true
tags: data-science python r vs-code 
---

In the past, I tried either call `Python` function from `R` environment
or call `R` functions from `Python`. I did not find good packages that 
could manage this well. I believe the best of way of leveraging 
the best of both Python and R is to share data via IO. 


## Best of both worlds

Many years ago, I spent nights and nights googling whether I should
learn R or Python first. Since I studied business and finance as my first
degree, R became my first choice. Later, I switched to Python when I 
studied mathematics and got into the world of machine learning. 

This year, a project I am working on needs me to use Python and R at the 
same time. I have already wrote a [post](https://oceanumeric.github.io/blog/data-science-environment){:target="_blank"} talking about setting up the
environment with `Docker`. In this post, I will share some tips about
how leveraged the best of both. 


## Pure scripts

Jupyter notebook used to be my top choice for doing some data analysis in
the quick way. However, I found it also distracting when you jump
among different code chunks and markdown blocks. It also become quite slow
once your iterations of analysis grows big. I think the most valuable 
thing that Jupyter provides is the kernel, which allows you:

- see results immediately
- render figures and tables 
- test a few lines of code quickly

Thanks to VS Code, we could run our code [interactively](https://code.visualstudio.com/docs/python/jupyter-support-py){:target="_blank"}. This gives me the choice of coding my thoughts with script purely but still keep analysis attached and flowed dynamically. 

Figure 1 to Figure 3 shows I mean with the above statement (please
zoom image to appreciate its powerful implications). 

<div class='figure'>
    <img src="/blog/images/vscode-setup1.png"
         alt="A demo figure to run python from command" class="zoom-img"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Programming 
        in the script and run code from the command line.
    </div>
</div>


<div class='figure'>
    <img src="/blog/images/vscode-setup2.png"
         alt="A demo figure run python interactively" class="zoom-img"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Programming 
        in the script and run code from interactive window with
        #%% option enabling code chunk. 
    </div>
</div>


<div class='figure'>
    <img src="/blog/images/vscode-setup3.png"
         alt="A demo figure run R interactively" class="zoom-img"
         style="width: 90%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> Programming in the 
        script with R and run the code from a Jupyter kernel interactively,
        which needs a shortcut setup (see the next section).
    </div>
</div>


## Pure scripts for R too 

To make the function demonstrated in Figure 3 work, you need setup 
a shortcut in VS Code. Just add the following keybindings setup
into your `keybindings.json`.


```
[
    {
        // I use shift and space to run 
        "key": "shift+space",
        // only run selection
        "command": "jupyter.execSelectionInteractive",
        // run whole file 
        // "command": "jupyter.runFileInteractive",
        "when": "editorTextFocus && editorLangId == 'r'"
    }
]
```


<div class='figure'>
    <img src="/blog/images/vscode-setup4.png"
         alt="A demo figure run R interactively" class="zoom-img"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Instructions for 
        finding the keyboard setup. 
    </div>
</div>

