---
title: To Compete or Not To Compete
subtitle: Living in a competitive world, it is very important to understand the pros and cons of competition. And more importantly, when to compete and when not to compete.
layout: blog_default
date: 2023-11-19
keywords: strategy, industry, competition, business, economics, game theory, philosophy, history, math
published: true
tags: economics game theory philosophy history
---

Ever since I want to create my own company, I have been thinking about having a 
strategy in this competitive world. For my case, I have to examine my situation
based on where I am, what kind of resources I have, and what are my relative
advantages and disadvantages, etc. All these factors lead me to think about
competition as I definitely want to avoid over-competition (or put it in
the Chinese context, to avoid '内卷', which is a very popular term in China).

Before we start, I want to spoil that this post is also a very good 
post on how to use big data to gain extremely valuable insights. It is also important to know beforehands that we are only focusing on the software industry in this post.


## Learning from Big Data

To aovid over-competition in a smart way, I simply ask myself a question: who
(or what kind of startups) survived and growed to be a decent size in Germany (as
I am currently living in Germany)? Here is how I tried to answer this question
step by step:

- Step 1: Gather Data from LinkedIn, such as the following table.

|  Index   | name                          | industry    |   found_date |   num_employees |
|:-----|:------------------------------|:---------------------|-------------:|:---------------------------:|
| 7914 | CONWIN Technologies GmbH | Software Development |         2002 |                          9 |
| 5743 | StoryBox GmbH            | Software Development |         2017 |                         25 |
| 5310 | Seneca Business Software GmbH | Software Development |         2011 |                          5 |
| 4236 | Lyvia Group                   | Software Development |       2020       |                        917 |
| 4725 | MyMinds                       | Software Development |         2021 |                          2 |


- Step 2: Clean the data and focus on German companies that were founed after 2016. 

<div class='figure'>
    <img src="/blog/images/num-startups-germany.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The number of firmes in Germany after 2016.
    </div>
</div>

As you can see from the Figure 1, the number of firms decreases when the year
decreases. This is because __Some firms exit the market and only a few firms
survive and grow to be a decent size.__ Even for those who survive, most of them
stay small unfortunately.


<div class='figure'>
    <img src="/blog/images/germany-firm-size.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Firm size distribution in Germany for those firms that were founded in 2015.
    </div>
</div>

Figure 2 shows the firm size distribution for those firms that were founded in 2015. As you can see, most of the firms are small and only a few firms grow to
be a decent size. 

- Step 3: Find the common features of those firms that survived and growed to be a decent size.

With the help of the data, I found the following common features of those firms
that survived and growed to be a decent size:

1. Firms who growed to be a decent size chose the right market that allows them to grow (no one can grow in a shrinking market).
2. Firms who growed to be a decent size clearly avoided over-competition, especially in the early stage from the US firms.
3. Almost all of the firms who survived focused on providing customized solutions to their customers with digital technologies.


## Stylzed Facts about Competition and Innovation

After checking the data from Linkedin, I also checked some literature about
competition and innovation. I found a very good paper called [The Schumpeterian Growth Paradigm](https://www.brown.edu/Departments/Economics/Faculty/Peter_Howitt/publication/Schumpeterian_Paradigm.pdf). In this paper, the authors found the following stylized facts about competition and innovation {% cite aghion2015schumpeterian %}:


- _Fact 1_: The relationship between competition and innovation follows an
inverted-U pattern, and the average technological gap within a sector increases with
competition.


<div class='figure'>
    <img src="/blog/images/neck-and-neck-competition.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The relationship between competition and innovation follows an inverted-U pattern; source: {% cite aghion2015schumpeterian %}
    </div>
</div>

In English, if two competitors are neck and neck, they are level with each other and __have an equal chance of winning__. Figure 3 shows that innovation drops when the competition is too low or too high. Furthermore, When firms are in neck-and-neck competition, they are more likely to innovate compared to the case when they are not in neck-and-neck competition.

What is the reason behind this fact? When firms have equal chance of winning, they are more likely to innovate to gain an advantage over their competitors. By innovating, they can not just differentiate themselves from their competitors, but also gain a technological advantage over their competitors if they succeed in innovating.

However, when firms are not having equal chance of winning, they are less likely to innovate. This is because the firms who are in the leading position are more likely to innovate to maintain their leading position. On the other hand, the firms who are in the lagging position are less likely to innovate because they are more likely to be eliminated from the market. Overall, the average technological gap within a sector increases with competition.

This explains why the term '内卷' is so popular in China. __With the higher competition, the average technological gap within a sector increases, which makes
those who are in the lagging position have less resources to innovate and
compete with those who are in the leading position.__

- _Fact 2_: More intense competition enhances innovation in frontier firms but may
discourage it in nonfrontier firms.

<div class='figure'>
    <img src="/blog/images/near-frontier.png"
         alt="euleriana_map"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> Total productiviy diverges for groups who are near frontier or far from; source: {% cite aghion2015schumpeterian %}
    </div>
</div>

Results from figure 4 are very striking as it shows that the total productivity
diverges so much between those who are near frontier and those who are far from
frontier with the increase of competition. This means __if you are not near frontier,
higher level of competition will even make you become less productive; however,
if you are near frontier, higher level of competition will make you become more
productive.__


Since individuals are judged by productivity, the total productivity divergence
for sure leads to the big income gap between those who are near frontier and
those who are far from frontier.


- _Fact 3_: Small firms exit more frequently. The ones that survive tend to grow
faster than average.

- _Fact 4_: A large fraction of R&D is done by incumbents.

- _Fact 5_: The growth of firms is highly correlated with the trustworthiness of
the legal system and the social norms of a country.


Both fact 3 and 4 are very intuitive, we will not discuss them here. Fact 5 is
very interesting as most people do not think about the importance of the
trustworthiness of the legal system and the social norms of a country. However,
it is very important to have a good legal system and social norms to support
all the business activities. 

The author used India as an example to show the importance of the trustworthiness of the legal system and the social norms of a country. In India, the legal system is not very good and the social norms are not very good either. As a result, the growth of firms in India is very slow because the firms in India only hire people who they know very well (such as family members, friends, etc.) to manage their firms. This contrains the growth of firms in India as the firms in India. 

<div class='figure'>
    <img src="/blog/images/firm-growh-indian.png"
         alt="firm growth in India"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> Firm growth trend in India, Mexcio and India source: {% cite aghion2015schumpeterian %}
    </div>
</div>

_Fact 6_: Average growth should decrease more rapidly as a country approaches
the world frontier when openness is low.

I listed fact 6 because this fact is very important for developing countries, especially for China. When China's economy grows to be near the world frontier, the average growth should decrease more rapidly if China does not open up its market. 

Fortunately, leaders in China are wise enough to know the importance of
opening up the market as it is shown in the following figure.

<div class='figure'>
    <img src="/blog/images/china-open-up.png"
         alt="firm growth in India"
         style="width: 87%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 6.</span> News about China's opening up the market.
    </div>
</div>

## Strategies for My Startup

After checking the data and the literature, I have the following strategies for
my startup:

- Carefully evaluate the level of competition in the market. If the level of
competition is too high, I will not enter the market. If the level of
competition is too low, I will not enter the market either.

- Carefully evaluate whether I am near frontier or far from frontier. If I am
near frontier, I will enter the market. If I am far from frontier, I will be very
careful to enter the market.

- Always avoid over-competition if I am not near frontier. 


- Focus on providing customized solutions to my customers with AI technologies.

Here I listed key business and technologies for my startup:

- Data Collection and Data Analysis
- Data Visualization
- Data Mining
- Software Development with Frontend and Backend
- Cloud Computing
- Machine Learning/Deep Learning

With the above dimensions, I measured my relative advantages and disadvantages
with two metrics: 1) to what extent those technologies have more space to
be customized and 2) to what extent I am near frontier in those technologies.

__Here is the answer generated by ChatGPT and me__:

| Technology                     | Customization Space | Proximity to Frontier |
|:---------------------------------|---------------------|------------------------|
| Data Collection and Analysis    | High                | High              |
| Data Visualization              | High                | High              |
| Machine Learning/Deep Learning  | Moderate            | High                |
| Software Development (Frontend) | High                | Moderate               |
| Data Mining                    | Moderate            | Moderate            |
| Software Development (Backend)  | High                | Low               |
| Cloud Computing                | Moderate            | Low                   |

I measue the proximity to frontier by checking how much I could understand when
I read the research papers in those technologies.

Since I realized that how much money you have to invest to train a large 
language model (such as GPT-3), I have already given up the idea of creating a
startup directly related to AI. However, I will focus on providing customized
solutions to my customers with AI technologies that are driven by data.
