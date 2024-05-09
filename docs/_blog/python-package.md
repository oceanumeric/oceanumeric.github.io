---
title: Python Package with Poetry
subtitle: Create a python package with poetry in production
layout: blog_default
date: 2024-05-09
keywords: python, poetry, package, pip, documentation, pypi, sphinx, pytest, coverage, codecov, black, flake8, mypy, pre-commit, git, github, gitlab, bitbucket, ci, cd, devops, production
published: true
tags: python poetry package pip pypi sphinx pytest cicd devops
---


Software enginners often feel that technology evolves at a rapid pace. From time to
time, we see new tools and libraries coming up. Many years ago, python developers
in the community were using `setuptools` and `pip` to create and distribute python
packages. But now, we have a new tool called `poetry` which is a modern tool for
dependency management and packaging in python. In this blog post, we will see how
to create a python package with poetry and publish it to pypi.

Here is a [demo package](https://github.com/oceanumeric/juskata){:target="_blank"} that I have created for a job interview. You can also `pip install juskata` to install the package for testing.

```python
# after installing the package
from juskata import Num2Words
# initialize the class
n2w_fr = Num2Words(lang="FR")
# convert a number to words
print(n2w_fr.convert_num(17))  # dix-sept
# convert a list of numbers to words
n2w_fr.convert_num_list([17, 80, 180000])
# ['dix-sept', 'quatre-vingts' 'cent-quatre-vingt-milles']
```

We will use `cookiecutter` to create a python package template. First, install `cookiecutter` using the following command.

```bash
python3 -m pip install --user cookiecutter
```

There are many [templates](https://www.cookiecutter.io/){:target="_blank"}, the one we will use is [`py-pkgs-cookiecutter`](https://github.com/py-pkgs/py-pkgs-cookiecutter){:target="_blank"}. There is also a [book](https://py-pkgs.org/){:target="_blank"} that you can read to learn more about creating python packages.

I put everything in the following pipeline. You can read the rest of the blog post to understand each step.

```bash
# Step 0: make sure you have pyenv installed
# Step 1: create a python package template
cookiecutter https://github.com/py-pkgs/py-pkgs-cookiecutter.git

# Step 2: initialize the git repository and setup the remote repository
cd justkata
git init
git add .
git commit -m "Initial commit for justkata package"

git remote add origin <remote-repository-url>
git branch -M main
git push -u origin main

# Step 3: install poetry and setup python environment (skip if you have poetry installed)
# linux, mac, windows(WSL)
curl -sSL https://install.python-poetry.org | python3 -
# after this, you need to restart your terminal
# check if poetry is installed
poetry --version

# install the python version you want to use
pyenv install 3.9  # or pyenv install 3.8.10 [whatever version you want]
# set the python version for the project
pyenv local 3.9
# create a virtual environment with poetry
poetry env use $(pyenv which python)
# activate the virtual environment
poetry shell

# you can check the list of virtual environments
poetry env list

# install the dependencies
peotry install

poeotry add black loguru jupyterlab --group dev

# Step 4: write the code and tests
# Step 5: manage the documentation with sphinx
poetry add --group dev myst-nb --python "^3.9"
poetry add --group dev sphinx-autoapi sphinx-rtd-theme

# Step 6: build the package and publish it to pypi
poetry build

# add the test pypi repository
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry publish -r test-pypi -u __token__ -p yourlongtestpypitokengoeshere...

# test the package
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
    justkata  # or whatever package name you have

# publish the package to the pypi server
poetry publish -u __token__ -p yourlongpypitokengoeshere...

# Step 7: check up the ci/cd pipeline
```


## 7 steps to create a python package with poetry

- Step 0: Setup the environment

I highly recommend using `pyenv` to manage your python versions. Please read [official documentation](https://github.com/pyenv/pyenv){:target="_blank"} to install `pyenv` on your machine.



- Step 1: Create a python package template using `cookiecutter`

```bash
cookiecutter https://github.com/py-pkgs/py-pkgs-cookiecutter.git
```

Like many `node.js` projects, you will be asked to provide some information about your package. For example, the package name, author name, email, etc.

```
author_name [Monty Python]: oceanumeric
package_name [mypkg]: justkata
package_short_description []: Convert numbers to words in French
package_version [0.1.0]: 0.1.0
python_version [3.9]: 3.9
Select open_source_license:
1 - MIT
2 - Apache License 2.0
3 - GNU General Public License v3.0
4 - Creative Commons Attribution 4.0
5 - BSD 3-Clause
6 - Proprietary
7 - None
Choose from 1, 2, 3, 4, 5, 6 [1]: 
Select include_github_actions:
1 - no
2 - ci
3 - ci+cd
Choose from 1, 2, 3 [1]:
```

- Step 2: Initialize the git repository and setup the remote repository

```bash
cd justkata
git init
git add .
git commit -m "Initial commit for justkata package"
```

If you are using github, you can create a new repository _without any README, LICENSE, or .gitignore_ files. Then, push the local repository to the remote repository.

```bash
git remote add origin <remote-repository-url>
git branch -M main
git push -u origin main
```

Your project structure should look like this at this point.

```
Current directory: juskata
.
├── CHANGELOG.md
├── CONDUCT.md
├── CONTRIBUTING.md
├── docs
├── LICENSE
├── pyproject.toml
├── README.md
├── src
└── tests
```

- Step 3: Install poetry and setup python environment

```bash
# linux, mac, windows(WSL)
curl -sSL https://install.python-poetry.org | python3 -
# after this, you need to restart your terminal
# check if poetry is installed
poetry --version
```

With poetry, you can manage the python environment and dependencies for the project. You can create a virtual environment with poetry and activate it.

```bash
# install the python version you want to use
pyenv install 3.9  # or pyenv install 3.8.10 [whatever version you want]
# set the python version for the project
pyenv local 3.9
# create a virtual environment with poetry
poetry env use $(pyenv which python)
# activate the virtual environment
poetry shell

# you can check the list of virtual environments
poetry env list
```

After setting up the python environment, you can install the dependencies for the project. With `pyproject.toml` file, poetry will install the dependencies for you.

```bash
poetry install
```

If you are not using any template, you can create a project with `poetry new <project-name>` and then add the dependencies to the `pyproject.toml` file.

```
poetry add justkata numpy pandas 
```

_Since we are developing a package, we should only add this for development._

```
poetry add black loguru jupyterlab --group dev
```

Those packages will be only used for development and testing purposes. This means
no one will install those packages when they install your package. In the following
example, the only dependency is python with version 3.9.

```bash
[tool.poetry.dependencies]
python = "^3.9"

[tool.poetry.dev-dependencies]

[tool.poetry.group.dev.dependencies]
pytest = "^8.2.0"
black = "^24.4.2"
pytest-cov = "^5.0.0"
jupyter = "^1.0.0"
myst-nb = {version = "^1.1.0", python = "^3.9"}
sphinx-autoapi = "^3.0.0"
sphinx-rtd-theme = "^2.0.0"
fastapi = "^0.111.0"
```

- Step 4: Write the code and tests

You can write the code in the `src` directory and tests in the `tests` directory. You can also use `pytest` to run the tests.

```bash
Current directory: src
.
├── juskata
│   ├── __init__.py
│   └── juskata.py
│   ├── # you can add more modules here, such as regression.py
```

__Please be careful with the naming of the directories and files.__ Within the `src` directory, you should have a directory with the same name as the package name. In this case, the package name is `juskata`. The `juskata` directory should have an `__init__.py` file. The module file does not have to be the same name as the package name. In this case, the module file is `juskata.py`.

__There is also a small trick to import the module in the `__init__.py` file.__

```python
# juskata/__init__.py
# read version from installed package
from importlib.metadata import version
from .juskata import *

__version__ = version("juskata")
```

With this trick, you can import the module in the `__init__.py` file and then import the package in the main script.

```python
from juskata import Num2Words
```

Without this trick, you would have to import the module directly in the main script.

```python
from juskata.juskata import Num2Words
```

If you are a user of the package, you do not want to see the package name twice!

When you finish writing the code and tests, you can run the tests with `pytest`.

```bash
# at root directory
pytest
```

You can also run the tests with coverage.

```bash
# at root directory
pytest --cov=juskata
```


- Step 5: Mangage the documentation with sphinx

You can use `sphinx` to manage the documentation for the package. You can install `sphinx` with poetry.

```bash
# here are the dependencies for documentation
poetry add --group dev myst-nb --python "^3.9"
poetry add --group dev sphinx-autoapi sphinx-rtd-theme
```

In `Vscode`, you can use [autodocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring){:target="_blank"} to generate the docstrings for the functions and classes automatically. This extension is using google style docstrings, which I highly recommend. Some people use numpy style docstrings, but I prefer google style docstrings as I think numpy style takes too much space.

Within the `docs` directory, there is an `example.ipynb` file that you can use to show how to use the package. Since we are using `myst-nb`, there is no need to add `rst` files. You can write the documentation in `markdown` format. Here is my documentation structure.

```bash
Current directory: docs
.
├── _build
├── changelog.md
├── conduct.md
├── conf.py
├── contributing.md
├── example.ipynb
├── images
├── index.md
├── make.bat
├── Makefile
├── __pycache__
└── requirements.txt
```

If you want to add more pages to the documentation, you can first create a markdown file and then add it to the `index.md` file.

```markdown
:maxdepth: 1
:hidden:

example.ipynb
changelog.md
contributing.md
add_more_pages.md
add_more_pages2.md
conduct.md
autoapi/index
``` 

We can build the documentation with the following command.

```bash
cd docs
make html
```

You can review the documentation if you have installed [live server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer){:target="_blank"} in `Vscode`. You can click on the `Go Live` button to see the documentation in the browser.


- Step 6: Build the package and publish it to pypi

You can build the package with the following command.

```bash
poetry build
```

Now assume you have registered an account on both [test.pypi.org](https://test.pypi.org/){:target="_blank"} and [pypi.org](https://pypi.org/){:target="_blank"}. You can publish the package to the test pypi server first.

```bash
# add the test pypi repository
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry publish -r test-pypi -u __token__ -p yourlongtestpypitokengoeshere...

# test the package
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
    justkata  # or whatever package name you have
```

If everything is fine, you can publish the package to the pypi server.

```bash
poetry publish -u __token__ -p yourlongpypitokengoeshere...
```

That's it! You have created a python package with poetry and published it to pypi. Now anyone can install your package with `pip install justkata`.


- Step 7: Check up the ci/cd pipeline

Go to your remote repository and check the ci/cd pipeline. The template we used has a github action workflow that runs the tests and builds the package. The documentation is also built and deployed to readthedocs. You can also add more steps to the workflow, such as deploying the package to pypi.



