---
title: "Using Stata with Jupyter Notebook in Linux System"
subtitle: Sometimes, you might need Stata 
layout: blog_default
date: 2022-10-19
keywords: Stata jupyter-notebook python 
published: true
tags: python stata jupyter 
---


It takes me a while to figure out how to use Stata with Jupyter Notebook
in Linux system. Just write down the procedures and others could do it 
by following those steps. 


## Download `.tar.gz` file

Please download your file first with your Stata serial number.

## Install 

- step 1: Change directory to wherever you have the `.tar.gz` file 
(e.g. the Downloads folder). There, create a temporary folder to 
store the installation files (e.g. statainstall), and extract 
the installation files to that folder.

```bash
cd ~/Downloads
mkdir statainstall
tar -xvzf Stata14Linux64.tar.gz -C statainstall
```

- step 2: Create the installation directory, and change location into it.

```bash
sudo mkdir /usr/local/stata16
cd /usr/local/stata16
```

- step 3: Run the install script.

```bash
sudo ~/Downloads/statainstall/install
```

## License

In order you configure the license file you just need to run `./stinit `
(you’ll need root privileges to write the file). Be sure to have the 
serial number, code and authorization. No need to disconnect from the 
interwebz here, even if you have an “alternative” license ;)

```bash
sudo ./stinit
```

## Add directory to path

Add execution path

```bash
echo export PATH="/usr/local/stata16:$PATH" >> ~/.bashrc
```

You need to source your .bashrc so that the changes are effective:


```bash
source ~/.bashrc
```

After adding the install directory to yout path, you should be able to run 
the appropriate version of Stata from the terminal, e.g.:


```bash
stata  # or stata-se or stata-mp (depends on your version)
```

You should see the following prompt.

<div class='figure'>
    <img src="/images/blog/stata.png"
         alt="A screenshot of stata"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Screenshot of Stata
    </div>
</div>

## Install stata_kernel 

Follow this [link](https://kylebarron.dev/stata_kernel/){:target="_blank"} and install
`stata_kernel`. 

```bash
pip install stata_kernel
python -m stata_kernel.install
```

Once you installed it, you might need to configure the path again (if you
are using `mp` version, then you are fine.). Since my stata license is a 
short-term usage on the SE version, I need to change the 
stata_path(default is MP version) in configuration file 
`~/.stata_kernel.conf` (use VS code `Ctr+Shift+P` to find and open it). 

```conf
[stata_kernel]
stata_path = /usr/local/stata16/stata-se  # change this one 
execution_mode = console
cache_directory = ~/.stata_kernel_cache
autocomplete_closing_symbol = False
graph_format = svg
graph_scale = 0.9  # change this one if you want smaller image 
user_graph_keywords = coefplot,vioplot
```
