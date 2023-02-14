---
title: "Making Animations for Mathematics"
subtitle: Based on the framework of 3brown1blue, I tried to make some animation for mathematics.  
layout: blog_default
date: 2022-11-03
keywords: math Manim animation python videos
published: true
tags: python math-animation
---

Suppose you want to create some animations like the following one by learning
an animation engine called [Manim](https://github.com/ManimCommunity/manim){:target="_blank"}.
However, the official document is difficult to follow. What should you do?

<iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In this series of posts, I will explain how to make animations step by step 
with `Manim`. To start it, let's install essential packages first. Please 
follow this [Link](https://docs.manim.community/en/stable/installation.html){:target="_blank"}
and install all packages. 


## Big picture 

A movie is made by different `scenes` and a scene is made by different kinds 
of `Mobject`s(moving objects). To create a scene we need to `create` or `add`
`Mobject`. 

Everything starts with the following `class`:

```py
class MovieName(Scene):
    def construct(self):
        # here you create your animation world
```

You could take `Scene` as your __camera__ or your background video which is waiting
for you to act. But who is the __actor__? In our engine, the actor is called
`Mobject` which is the most important element (we call it `base class`). 
The following figure gives the big picture. 

<div class='figure'>
    <img src="/images/blog/manim.png"
         alt="A big picture"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Big Picture of Manim
    </div>
</div>


Every time when you want to create a movie, you need to do the following
three steps:

1. set up the `Scene`
2. create the `Mobject`
3. call animation `method` to make `Mobject` to move

Sometimes, you might need to zoom in or zoom out with your `camera`, which
we will cover those advanced topics in detail later. 

## Set up the `Scene` 

To set up the scene, you just need to import the `manim` package and create
an inherited class from `Scene`. 

```py
from manim import *   # import 


class YourVideoName(Scene):
    def construct(self):
        # your code starts here 
```

## Create the `Mobject`

A _moving object_ is a base class that could engine your animation. For instance,
a simple animation is to make a photo fly in from the left to the right. Then,
a photo is a _moving object_. In `manim`, `ImageMobject` could read an image
and make it as a moving object. 

```python
img = ImageMobject("images/cuteguy1.png")
img.scale(0.8)
img.shift(LEFT*5)
self.play(img.animate.shift(RIGHT*5), run_time=2)
```

There are many _moving objects_ in `manim`, here is the link:

* `frame`: Special rectangles.
* `geometry` - Various geometric Mobjects.
* `graph` - Mobjects used to represent mathematical graphs (think graph theory, not plotting).
* `graphing` - Coordinate systems and function graphing related mobjects.
* `logo` - either the logo from the package or created by yourself (Utilities for Manim's logo and banner).
* `matrix` - Mobjects representing matrices.
* `mobject` -  Base classes for objects that can be displayed.
* `svg` - Mobjects related to SVG images.
* `table` -  Mobjects representing tables.
* `text` - Mobjects used to display Text using Pango or LaTeX.
* `three_d` - Three-dimensional mobjects.
* `types` - Specialized mobject base classes.
* `value_tracker` - Simple mobjects that can be used for storing (and updating) a value.
* `vector_field` - Mobjects representing vector fields.

## Call animation `method` 

Once you created the `Mobject`, then you can call `animation` methods to make
them animated. For instance, we have already showed a simple way to animate 
an `Mojbect`:

```py
self.play(img.animate.shift(RIGHT*5), run_time=2)
```

Here are the full list of `animation` methods:

* `animation`: Animate mobjects.
* `changing`: Animation of a mobject boundary and tracing of points.
* `composition`: Tools for displaying multiple animations at once.
* `creation`: Animate the display or removal of a mobject from a scene.
* `fading`: Fading in and out of view.
* `growing`: Animations that introduce mobjects to scene by growing them from points.
* `indication`:Animations drawing attention to particular mobjects.
* `movement`: Animations related to movement.
* `numbers`: Animations for changing numbers.
* `rotation`: Animations related to rotation.
* `specialized`: 
* `speedmodifier`: Utilities for modifying the speed at which animations are played.
* `transform`: Animations transforming one mobject into another.
* `transform_matching_parts`: Animations that try to transform Mobjects while keeping track of identical parts.
* `updaters`:Animations and utility mobjects related to update functions.


Each __animation__ method has its functions and you need to refer to the document
to figure out how to use them. For instance, if you want to `fading` some `Mobject`,
you can do:

```py
from manim import *

class Fading(Scene):
    def construct(self):
        tex_in = Tex("Fade", "In").scale(3)
        tex_out = Tex("Fade", "Out").scale(3)
        self.play(FadeIn(tex_in, shift=DOWN, scale=0.66))
        self.play(ReplacementTransform(tex_in, tex_out))
        self.play(FadeOut(tex_out, shift=DOWN * 2, scale=1.5))
```

## Examples


Every time you want to create a video with `manim`, you need to think about

* what kind of `Mobject` do you need?
* what kind of attributes of `Mobject` do you need?
* how can you animate those `Mobject` by calling different kinds of animation
methods 

<div class='figure'>
    <img src="/images/blog/manim2.png"
         alt="A big picture"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> Big Picture of Manim
    </div>
</div>


## Gradient descent visualization: version 1

To visualize the gradient descent process, we need to the following `Mobject`:

* `graphing` - `coordinate_systems` - `Axes` 
* `text` - `numbers` - `Variable` 

To animate the object, we could use:

* `animate`
* `add_updater`

<iframe width="560" height="315" src="https://www.youtube.com/embed/X1Ma4JK-YV4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

To learn how to use `Axes`, you can either read the [document](https://docs.manim.community/en/stable/reference/manim.mobject.graphing.coordinate_systems.Axes.html#axes){:target="_blank"} or read the [source code](https://docs.manim.community/en/stable/_modules/manim/mobject/graphing/coordinate_systems.html#Axes){:target="_blank"}. I think
reading `source code` is more efficient as it gives you every parameter and 
method you can utilize for this `Axes` class. 

Here is the code:

```py
from pickle import NONE
from manim import *   # import all packages from manim
import numpy as np
import matplotlib.pyplot as plt


class GradientDescent(Scene):
    
    def construct(self):
        # construct the axes
        ax = Axes(
            x_range=[-4, 6],
            y_range=[-6, 6]
        ).add_coordinates()
        
        # plot the function
        def func(x):
            return x**2 - 2*x - 3
        func_graph = ax.plot(func, color=BLUE)
        
        # add function label
        func_label = ax.get_graph_label(
            func_graph, "f(x) = x^2-2x -3", x_val=-4, direction=UP
        )
        # add a vertical line of the minimum value
        line_1 = ax.get_vertical_line(ax.i2gp(1, func_graph), color=YELLOW)
        
        # derivate function and plot 
        def derivate(x):
            return 2*x - 2
        derivate_graph = ax.plot(derivate, color="#F8A331")
        derivate_label = ax.get_graph_label(
            derivate_graph, "f'(x) = 2x-2", x_val=-3, direction=UP*6
        )
        
        # gradient descent variable with alpha = 0.03, iteration = 50
        # iteration tracker
        iteration = Variable(1, Text("Iteration"), num_decimal_places=0).scale(0.8)
        
        # learning rate
        learning_rate = 0.97
        learning_rate_var = Variable(learning_rate, 
                                     MathTex(r"\alpha"), 
                                     num_decimal_places=2).shift(UP*3.2+LEFT*0.3)
        learning_rate_var.value.set_color(RED)
        
        # theta 
        theta = 4
        theta_var = Variable(theta, r"\theta_t", num_decimal_places=3).scale(0.8)
        theta_var.value.set_color(GREEN)
        theta_var_update = Variable(theta, r"\theta_{t+1}", num_decimal_places=3).scale(0.8)
        theta_var_update.value.set_color(GREEN)
        theta_var.shift(DOWN*2.7+RIGHT*3.4)
        theta_var_update.shift(DOWN*3.3+RIGHT*2)
        
        # function
        def constant_one(x):
            if x != 1:
                return 1
            else:
                return 1
        # update theta_var
        theta_var.add_updater(lambda x: x.tracker.set_value(
            (theta_var.tracker.get_value()-
             learning_rate*constant_one(iteration.tracker.get_value())*
             derivate(theta_var.tracker.get_value()))
        )) 
        
        # update theta_var_update
        theta_var_update.add_updater(lambda x: x.tracker.set_value(
            (theta_var.tracker.get_value()-
             learning_rate*constant_one(iteration.tracker.get_value())*
             derivate(theta_var.tracker.get_value()))
        ))
        
        # add gradient descent formula 
        update_rule = MathTex(r"\theta_{t+1} = \theta_t - \alpha f'(x)")
        iter_update_group = Group(iteration, update_rule).arrange(DOWN).shift(DOWN*1.7+RIGHT*3.9)
        
        # add moving point
        mp = theta_var_update.tracker
        mp_initial_point = [ax.coords_to_point(mp.get_value(), func(mp.get_value()))]
        mp_dot = Dot(point=mp_initial_point, color=BLUE, fill_opacity=0.7,
                     radius=0.15)
        mp_dot.add_updater(lambda x: x.move_to(ax.c2p(
            mp.get_value(),
            func(mp.get_value())
        )))
        # add theta moving point
        mp_theta_init = [ax.c2p(mp.get_value(), 0)]
        mp_theta_dot = Dot(point=mp_theta_init, color=GREEN, radius=0.15)
        mp_theta_dot.add_updater(lambda x: x.move_to(ax.c2p(
            mp.get_value(),
            0
        )))
        
        # add derivative variable
        derivate_var = Variable(
            derivate(theta_var.tracker.get_value()),
            r"f'(x)",
            num_decimal_places=3
        ).scale(0.8).shift(DOWN*1.5+LEFT*6)
        derivate_var.label.set_color("#F8A331")
        derivate_var.value.set_color("#F8A331")
        
        derivate_var.add_updater(lambda x: x.tracker.set_value(
            derivate(mp.get_value())
        ))
        
        # add function variable
        func_var = Variable(
            func(mp.get_value()),
            r"f(x)",
            num_decimal_places=3
        ).scale(0.8).shift(UP*2.5+LEFT*6.4)
        func_var.label.set_color(BLUE)
        func_var.value.set_color(BLUE)
        
        func_var.add_updater(lambda x: x.tracker.set_value(
            func(mp.get_value())
        ))
        
        # add all objects 
        self.add(ax, func_graph, func_label, line_1,
                 mp_dot, mp_theta_dot,
                 derivate_graph, derivate_label,
                 learning_rate_var,
                 iter_update_group,
                 func_var, derivate_var,
                 theta_var, theta_var_update)
        
        
        self.play(iteration.tracker.animate.set_value(20), run_time=13)
        self.wait()
```

## Gradient descent visualization: version 2

when you watch the first video, you can notice that the iteration and $\theta$ 
are not updating with the same rate. This happens due to the `add_updater` function,
which is defined as:

```py
class Scene:
    """A Scene is the canvas of your animation.

    The primary role of :class:`Scene` is to provide the user with tools to 
    manage mobjects and animations.  Generally speaking, a manim script 
    consists of a class that derives from :class:`Scene` 
    whose :meth:`Scene.construct` method is overridden by the user's code.
    """ 
    # ...
    # ...
    def add_updater(self, func: Callable[[float], None]) -> None:
        """Add an update function to the scene.

        The scene updater functions are run every frame,
        and they are the last type of updaters to run.
        """
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/VLlRqUK8ffA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The above video works well as the iteration and $\theta$ are updating at
the same rate. This version uses `always_redraw` function which makes each
frame to update with the same value of `iteration`. However it takes much more
time to create the video as it needs to create more scenes. 

Here is the source code:

```py
from pickle import NONE
from manim import *   # import all packages from manim
import numpy as np
import matplotlib.pyplot as plt


class GradientDescent(Scene):
    
    def construct(self):
        
        # add Axes object
        ax = Axes(
            x_range=[-4, 6],
            y_range=[-6, 6]
        ).add_coordinates()
        
        # plot the function
        def func(x):
            return x**2 - 2*x - 3
        func_graph = ax.plot(func, color=BLUE)
        
        # add function label
        func_label = ax.get_graph_label(
            func_graph, "f(x) = x^2-2x -3", x_val=-4, direction=UP
        )
        # add a vertical line of the minimum value
        line_1 = ax.get_vertical_line(ax.i2gp(1, func_graph), color=YELLOW)
        
        # derivate function and plot 
        def derivate(x):
            return 2*x - 2
        derivate_graph = ax.plot(derivate, color="#F8A331")
        derivate_label = ax.get_graph_label(
            derivate_graph, "f'(x) = 2x-2", x_val=-3, direction=UP*6
        )
        
        # gradient descent variable with alpha = 0.03, iteration = 50
        
        # add learning rate on the top of scene
        learning_rate = 0.97
        learning_rate_var = Variable(learning_rate, 
                                     MathTex(r"\alpha"), 
                                     num_decimal_places=2).shift(UP*3.2+LEFT*0.3)
        learning_rate_var.value.set_color(RED)
        
        # iteration tracker
        iteration = ValueTracker(1)  # start from 1 
        
        # add iteration text 
        iteration_text = (
            Text("Iteration: ").scale(0.8)
            .shift(DOWN*1.3+RIGHT*3.7)
        )
        iteration_value = always_redraw(
            lambda: DecimalNumber(num_decimal_places=0)
            .set_value(iteration.get_value())
            .next_to(iteration_text, RIGHT, buff=0.2).shift(DOWN*0.03)
        )
        
        # add gradient descent formula below the iteration
        update_rule = (
            MathTex(r"\theta_{t+1} = \theta_t - \alpha f'(x)")
            .shift(DOWN*2+RIGHT*3.9)
        )
                    
        # add theta and theta_update
        theta_text = (
            MathTex(r"\theta_t = ").scale(0.8)
            .shift(DOWN*2.6+RIGHT*3.9)
        )
        # initialize the theta value
        initial_theta = 4.0
        
        # helper function
        def update_theta_t(initial_value, tracker_value):
            theta = initial_value
            for i in range(int(tracker_value)-1):
                theta = theta - learning_rate * derivate(theta)
            return theta
        
        theta_value = always_redraw(
            lambda: DecimalNumber(num_decimal_places=3)
            .set_value(
                update_theta_t(initial_theta, iteration.get_value())
            )
            .next_to(theta_text, RIGHT, buff=0.1).shift(DOWN*0.01)
            .set_color(GREEN)
            .scale(0.8)
        )
        
        # add theta_t+1
        theta_update_text = (
            MathTex(r"\theta_{t+1} = ").scale(0.8)
            .shift(DOWN*3.2+RIGHT*2.7)
        )
        
        theta_update_value = always_redraw(
            lambda: DecimalNumber(num_decimal_places=3)
            .set_value(
                update_theta_t(initial_theta, iteration.get_value()+1)
            )
            .next_to(theta_update_text, RIGHT, buff=0.1).shift(UP*0.01)
            .set_color(GREEN)
            .scale(0.8)
        )
        
        # add derivative value
        derivate_text = (
            MathTex(r"f'(x) = ").scale(0.8)
            .shift(DOWN*1.5+LEFT*5.6)
            .set_color("#F8A331")
        )
        
        def update_derivate_value(tracker_value):
            x = update_theta_t(initial_theta, tracker_value+1)
            return derivate(x)
            
        derivate_text_value = always_redraw(
            lambda: DecimalNumber(num_decimal_places=3)
            .set_value(
                update_derivate_value(iteration.get_value())
            )
            .next_to(derivate_text, RIGHT, buff=0.1).shift(UP*0.01)
            .set_color("#F8A331")
            .scale(0.8)
        )
        
        # add function value
        func_text = (
            MathTex(r"f(x) = ").scale(0.8)
            .shift(UP*2.5+LEFT*5.9)
            .set_color(BLUE)
        )
        
        def update_func_value(tracker_value):
            x = update_theta_t(initial_theta, tracker_value+1)
            return func(x)
        
        func_text_value = always_redraw(
            lambda: DecimalNumber(num_decimal_places=3)
            .set_value(
                update_func_value(iteration.get_value())
            )
            .next_to(func_text, RIGHT, buff=0.1).shift(UP*0.01)
            .set_color(BLUE)
            .scale(0.8)
        )
        
        # add moving point on graph
        # helper function
        def mp_coordinate(tracker_value):
            x = update_theta_t(initial_theta, tracker_value+1)
            y = func(x)
            return x, y
        
        mp0 = always_redraw(
            lambda: Dot(
                point=[ax.c2p(theta_value.get_value(), func(theta_value.get_value()))],
                color=BLUE,
                fill_opacity=0.8, radius=0.15
                ).move_to(
                    ax.c2p(
                        mp_coordinate(iteration.get_value())[0],
                        mp_coordinate(iteration.get_value())[1]
                    )
                )
        )
        
        mp1 = always_redraw(
            lambda: Dot(
                point=[ax.c2p(theta_value.get_value(), func(theta_value.get_value()))],
                color=GREEN,
                fill_opacity=0.8, radius=0.15
                ).move_to(
                    ax.c2p(
                        mp_coordinate(iteration.get_value())[0],
                        0
                    )
                )
        )
        
        self.add(ax, func_graph, func_label, line_1,
                 derivate_graph, derivate_label,
                 learning_rate_var,
                 iteration_text, iteration_value,
                 update_rule,
                 theta_text, theta_value,
                 theta_update_text, theta_update_value,
                 derivate_text, derivate_text_value,
                 func_text, func_text_value,
                 mp0, mp1)
        self.play(iteration.animate.set_value(20), run_time=13, rate_func=linear)
        self.wait()
```