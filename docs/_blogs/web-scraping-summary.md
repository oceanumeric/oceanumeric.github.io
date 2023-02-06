---
title: "Web Scraping: A Summary"
subtitle: After scraped a fair amount of websites, I wrote down what I have learned. 
layout: blog_default
date: 2022-10-18
keywords: webscraping python javascript
published: true
tags: webscraping python javascript
---


There are many ways to do web scrapping. In this post, I will summarize
them into three categories and give methods to scrape those different websites:

* [methods for static websites](#methods-for-static-websites)
* [methods for dynamic websites](#methods-for-dynamic-websites)
* [methods for single-page application websites](#methods-for-single-page-application-websites) 

## Methods for static websites

If the website you want to scrape is a static one, you could just use `requests`
and `BeautifulSoup`. The starting function could be like the following one.

```python
from bs4 import BeautifulSoup
import requests


def url_to_soup(url, headers = HEADERS):
    try:
        # you may turn verify to False might unblock multiple requests
        page = requests.get(url,  headers=headers, verify=True)
        print(page)
        page.encoding = "utf-8"
        print(page.status_code)
        soup = BeautifulSoup(page.content, "lxml")
        return soup
    except requests.exceptions.ConnectionError:
        print("Connection refused: "+url)
```

## Methods for dynamic websites

If the website you want to scrape is a dynamic one, you need to inspect 
`Network` via `Developer Tools` in your browser and figure out how the data
was rendered dynamically. Depending on how the data is rendered, you could try:

* copy `cURL` and request it in [postman](https://web.postman.co/){:target="_blank"}, in which
`Code snippet` could be generated automatically by `postman` for you
* combine `requests` with [Splash](https://splash.readthedocs.io/en/stable/){:target="_blank"} - a javascript rendering service. 
* using [Selenium](https://www.selenium.dev/){:target="_blank"}

I prefer using `Splash` as it allows you to render javascript in `Python`
environment. For instance, 

```py
res = requests.get("http://localhost:8050/render.html",
                            params={"url": url, "wait": 2})
soup = BeautifulSoup(res.content, "html.parser")
```

## Methods for single-page application websites 

For the single-page application websites, you need to write a `javascript`
script and add it into `Sources` of developer tools of your browser as it
is shown in the following figure.

<div class='figure'>
    <img src="/images/blog/snippet.png"
         alt="A screenshot of developer tools"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Screenshot of Developer Tool
    </div>
</div>


In this example, the function `getData` was rendered from the website we want 
to scrape. Once it was rendered, then we could call it directly from `Console`
and get the data we want. 

After that, you could use the following function to save your dataset.

```js
(function(console){
    // define a function 
    console.save = function(data, filename){

        if(!data) {
            console.error('Console.save: No data')
            return;
        }
        // file name = 'webdata.json' 
        if(!filename) filename = 'webdata.json'

        if(typeof data === "object"){
            data = JSON.stringify(data, undefined, 4)
        }
        // create a blob object to save data 
        var blob = new Blob([data], {type: 'text/json'}),
            e    = document.createEvent('MouseEvents'),
            a    = document.createElement('a')

        a.download = filename
        a.href = window.URL.createObjectURL(blob)
        a.dataset.downloadurl =  ['text/json', 
                                    a.download, a.href].join(':')
        e.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, 
                            false, false, false, false, 0, null)
        a.dispatchEvent(e)
    }
})(console)
```

Every time you run `console.save([your_data])`, your browser will automatically
download the data into a `json` file called `webdata.json`. 

## Javascript framework

If you are seeking for a Javascript framework that provides a high-level API to 
control Chrome or Chromium over the DevTools Protocol, you can use [Puppeteer](https://pptr.dev/){:target="_blank"}.

__References:__
<br>

[1] [DevTools Snippets](http://bgrins.github.io/devtools-snippets/#console-save){:target="_blank"}<br>
[2] [Web APIs Blob](https://developer.mozilla.org/en-US/docs/Web/API/Blob){:target="_blank"}<br>
[3] [Scraping XHR](https://scrapism.lav.io/scraping-xhr/){:target="_blank"}<br>
