---
title: "Making Videos with Manim"
subtitle: Manim could be used to make videos for non-mathematical purposes as well.
layout: blog_default
date: 2023-07-20
keywords: math Manim animation python videos
published: true
tags: python math-animation education
---

Manim could be used to make videos for non-mathematical purposes as well. In this post,
I will try to use Manim to replicate a video from `ByteByteGo`. Here is the video:

<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-mN3VyJuCjM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

When I try to replicate this video, I want to decompose the video into many
reusable parts. For instance, the introduction part should be same for 
all the videos. The goal is to write resuable code so that I can make
many videos with less effort.