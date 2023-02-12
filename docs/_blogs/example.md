---
title: An Example Post
subtitle: Here is some extra detail about the post.
layout: blog_default
date: 2022-10-17
keywords: blogging, writing
published: true
tags: jekyll
---


## Informal justifications

{% katexmm %}

In _Auto-Encoding Variational Bayes_, {% cite kingma2013auto %}, Kingma presents an
unbiased, differentiable, and scalable estimator for the ELBO in variational inference.
A key idea behind this estimator is the _reparameterization trick_. But why do we need 
this trick in the first place? When first learning about variational autoencoders 
(VAEs), I tried to find an answer online but found the explanations too informal...


$$
e = mc^2. \tag{1}
$$


Cool!

```python
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import os
import sys


# check executable path
print(sys.executable)
```

There are something: 

$$
\mathbb{P}(X=2) = \frac{1}{3}
$$

wish you can do 

```html
{% raw %}
<div id='posts' class='section'>
    {%for post in site.posts%}
        <div class='post-row'>
            <p class='post-title'>
                <a href="{{ post.url }}">
                    {{ post.title }}
                </a>
            </p>
            <p class='post-date'>
                {{ post.date | date_to_long_string }}
            </p>
        </div>
        <p class='post-subtitle'>
            {{ post.subtitle }}
        </p>
    {% endfor %}
</div>
{% endraw %}
```

### Citations

According to {% cite bishop2006pattern %}, machine learning ... You can
cite another one {% cite vaswani2017attention %}


### Figures

#### Without zoom 


<div class='figure'>
    <img src="/images/inclusion-exclusion.png"
         alt="A demo figure"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Diagram of rejection sampling. The 
        density $q(\mathbf{z})$ must be always greater than $p(\mathbf{x})$. A new sample 
        is rejected if it falls in the gray region and accepted otherwise. These accepted 
        samples are distributed according to $p(\mathbf{x})$. This is achieved by sampling 
        $z_i$ from $q(\mathbf{z})$, and then sampling uniformly from $[0, k q(z_i)]$. 
        Samples under the curve $p(z_i)$ are accepted.
    </div>
</div>


#### With zoom 

<div class='figure'>
    <img src="/images/blog/oecd-han-database.png"
         alt="OECD HAN database illustration" class="zoom-img"
         style="width:100%"
         />
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> OECD HAN database structure:
        there are four tables, in which HAN_PERSON is the correspondence
        table which include all cleaned names.
    </div>
</div>

### Tables


<div class="table-caption">
<span class="table-caption-label">Table 1.</span> First six rows of HAN_PATENTS, which starts at HAN_ID = 4. 
</div>

| HAN_ID <int> | HARM_ID <int> | Appln_id <int> | Publn_auth <chr> | Patent_number <chr> |
|--------------|---------------|----------------|------------------|---------------------|
| __4__           | 4             | 311606173      | US               | __US8668089__          |
| 7            | 7             | 439191607      | US               | US9409947           |
| 7            | 7             | 518367793      | US               | US10836794          |
| 10           | 10            | 365204276      | US               | US8513480           |
| 14           | 14            | 336903179      | WO               | WO2011112122        |
| 14           | 14            | 363622722      | WO               | WO2012064218        |


### Table over flow if you need it

<div class="table-wrapper" markdown="block">

| Analysis 	| Patent statistics 	| Purposes 	|
|:---	|:---|:---	|
| Citation analysis 	| Forward citations 	| Measure technological value 	|
|  	| Backward citations 	| Find knowledge source 	|
| Patent counts  analysis	| Patent counts 	| Observe patent portfolio 	|
| 	| RTA (Revealed Technology Advance) 	| Identify core technological competence 	|
| | PS (Patent Share) |  |
| Technology class analysis 	| Generality 	| Measure endogenous applicability to different technological fields 	|
|  	| Originality 	| Measure knowledge absorption from different technological fields 	|
| Inventor analysis 	| Inventor counts 	| Measure invention quality  	|
| | | Measure absorptive capability|
|  	| Inventor 	| Identify specific inventors' info such as star engineers 	|
| | | Follow mobility of R&D personnel | 

</div>

{% endkatexmm %}
