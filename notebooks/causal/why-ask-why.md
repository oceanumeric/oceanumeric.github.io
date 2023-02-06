# Why ask why? 


Following [Gelman (2011)](https://arxiv.org/pdf/1003.2619.pdf), 
we distinguish two broad classes of causal queries:

1. _Forward causal questions_, or the estimation of “effects of causes.” What might happen
if we do X? What is the effect of some manipulation, e.g., the effect of smoking on
health, the effect of schooling on earnings, the effect of campaigns on election outcomes,
and so forth?

2. _Reverse causal inference_, or the search for “causes of effects.” What causes Y? 
Why do more attractive people earn more money? Why does per capita income vary some
much by country? do many poor people vote for Republicans and rich people vote for
Democrats? Why did the economy collapse?

In forward reasoning, the potential treatments under study are chosen 
ahead of time, whereas, in reverse reasoning, the research goal is to find 
and assess the importance of the causes. The distinction between forward and 
reverse reasoning (also called “the effects of causes” and the 
“causes of effects”) was made by Mill (1843). Forward causation is a pretty 
clearly-defined problem, and there is a consensus that it can be modeled using 
the counterfactual or potential outcome notation associated with Neyman (1923) 
and Rubin (1974) and expressed using graphical models by Pearl (2009): the 
causal effect of a treatment T on an outcome Y for an individual person (say), 
is a comparison between the value of Y that would’ve been observed had 
the person followed the treatment, versus the value that would’ve been 
observed under the control; in many contexts, the treatment effect for person 
i is defined as the difference $Y_i(T=1) - Y_i(T=0)$. Many common techniques, 
such as differences in differences, linear regression, and instrumental variables,
can be viewed as estimated average causal effects under this definition. 

hen statistical and econometric methodologists write about causal inference, 
they generally focus on forward causal questions. We are taught to answer 
questions of the type “What if?”, rather than “Why?” Following the work by 
Rubin (1977) causal questions are typically framed in terms of manipulations: 
if $x$ were changed by one unit, how much would $y$ be expected to change? But 
reverse causal questions are important too. They are a natural
way to think (consider the importance of the word Why, or see Sloman, 2005, 
for further discussion of causal ideas in cognition). In many ways, it is 
the reverse causal questions that motivate the research, including experiments 
and observational studies, that we use to answer the forward questions.

The question discussed in the current paper is: How can we incorporate reverse causal
questions into a statistical framework that is centered around forward causal inference?
Our resolution is as follows: __Forward causal inference is about estimation; 
reverse causal questions are about model checking and hypothesis generation__. 
To put it another way, we ask reverse causal questions all the time, 
but we do not perform reverse causal inference. We do not try to answer 
“Why” questions; rather, “Why” questions motivate “What if” questions that 
can be studied using standard statistical tools such as experiments, 
observational studies, and structural equation models. 

## Conditional probability

Every time we ask _what if_ question, we ask within our way of thinking and framing,
which means that our question is always __conditional__ on something. Therefore,
you need to have a mindset of contingency after all everything depends on
everything. 

Given two events $A$ and $B$ from the sigma-field of a probability space, 
with the unconditional probability of $B$ being greater than zero
 (i.e., $\mathrm{P}(B)>0)$, the conditional probability of $A$ 
 given $B(P(A \mid B))$ is the probability of $A$ occurring if $B$ has or 
 is assumed to have happened. ${ }^{[5]} A$ is assumed to a set of all possible 
 outcomes of an experiment or random trial that has a restricted or 
 reduced sample space. The conditional probability can be found by 
 the quotient of the probability of the joint intersection of events 
 $A$ and $B(P(A \cap B))$-the probability at which $A$ and $B$ occur together, 
 although not necessarily occurring at the same time-and the probability of $B:

$$
P(A \mid B)=\frac{P(A \cap B)}{P(B)}
$$

## Conditional expectation

### Conditioning on an event

If $A$ is an event in $\mathcal{F}$ with nonzero probability, and $X$ is a 
discrete random variable, the conditional expectation of $X$ given $A$ is

$$
\begin{aligned}
\mathrm{E}(X \mid A) &=\sum_x x P(X=x \mid A) \\
&=\sum_x x \frac{P(\{X=x\} \cap A)}{P(A)}
\end{aligned}
$$

where the sum is taken over all possible outcomes of $X$.
Note that if $P(A)=0$, the conditional expectation is undefined due to the division by zero.

### Discrete random variables

If $X$ and $Y$ are discrete random variables, the conditional expectation of $X$ given $Y$ is
$$
\begin{aligned}
\mathrm{E}(X \mid Y=y) &=\sum_x x P(X=x \mid Y=y) \\
&=\sum_x x \frac{P(X=x, Y=y)}{P(Y=y)}
\end{aligned}
$$
where $P(X=x, Y=y)$ is the joint probability mass function of $X$ and $Y$. 
The sum is taken over all possible outcomes of $X$.
Note that conditioning on a discrete random variable is the same as 
conditioning on the corresponding event:

$$
\mathrm{E}(X \mid Y=y)=\mathrm{E}(X \mid A)
$$
where $A$ is the set $\{Y=y\}$.

### Continuous random variables

Let $X$ and $Y$ be continuous random variables with joint density 
$f_{X, Y}(x, y), Y$ 's density $f_Y(y)$, and conditional density 
$f_{X \mid Y}(x \mid y)=\frac{f_{X, Y}(x, y)}{f_Y(y)}$ of $X$ given 
the event $Y=y$. The conditional expectation of $X$ given $Y=y$ is

$$
\begin{aligned}
\mathrm{E}(X \mid Y=y) &=\int_{-\infty}^{\infty} x f_{X \mid Y}(x \mid y) \mathrm{d} x \\
&=\frac{1}{f_Y(y)} \int_{-\infty}^{\infty} x f_{X, Y}(x, y) \mathrm{d} x
\end{aligned}
$$

Reference: [Why ask why?
Forward causal inference and reverse causal questions](http://www.stat.columbia.edu/~gelman/research/unpublished/reversecausal_13oct05.pdf)