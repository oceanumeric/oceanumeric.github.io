# ChatGPT Demo

This is a demo of the ChatGPT, which I will show you how powerful it is. I will use it to show how you can do research with Github Copilot.


## Logic reasoning with Github Copilot


I will use a probability model to show how Github Copilot can do logic reasoning. The probability model is as follows. Suppose we have a coin, and we flip it three times. What's the probability that we get three heads? Let's this is a fair coin, and each flip is independent. Then the probability is 1/8. Here is how we could calculate it using Github Copilot.

$$
\begin{align}
P(\text{three heads}) &= P(\text{head}) \times P(\text{head}) \times P(\text{head}) \\
&= \frac{1}{2} \times \frac{1}{2} \times \frac{1}{2} \\
&= \frac{1}{8}
\end{align}
$$

Now, let's use python to plot the probability distribution of the number of heads we get. We will flip the coin 1000 times, and plot the probability of getting 0, 1, 2, or 3 heads.

```python
import numpy as np
import matplotlib.pyplot as plt


