---
title: Approximating the Posterior
subtitle: When we use Bayesian inference, we need to compute the posterior distribution. In this post, we will look at some methods for approximating the posterior distribution.
layout: math_page_template
date: 2023-04-12
keywords: probabilistic-thinking machine-learning bayesian-inference bayesian-statistics conjugate-prior conjugate-family
published: true
tags: probability algorithm data-science machine-learning binomial-distribution bayesian-statistics beta-distribution conjugate-prior normal-distribution
---

In previous posts, we have looked at the Bayesian inference framework. In this post, we will look at some methods for approximating the posterior distribution. 

## Grid approximation

{% katexmm %}

With the beta-binomial bayesian model, we can update the posterior distribution by using the conjugate prior. However, in general, we cannot use the conjugate prior to update the posterior distribution. In this case, we need to use numerical methods to approximate the posterior distribution. One of the simplest methods is the grid approximation.

For instance, assume we have the following model:

$$
\begin{aligned}
Y | \theta &\sim \text{Binomial}(n, \theta) \\
\ \ \theta &\sim \text{Beta}(\alpha = 2, \beta = 2)
\end{aligned}
$$

We can interpret $Y$ as the number of successes in $n$ trials, and $\theta$ as the probability of success in each trial. Suppose we observe $Y = 9$, we can use beta-binomial bayesian model to update the posterior distribution:

$$
\begin{aligned}
\theta | Y = 9 &\sim \text{Beta}(\alpha + 9, \beta + n - 9) \\
&= \text{Beta}(11, 3)
\end{aligned}
$$

However, in general, we cannot use the conjugate prior to update the posterior distribution. In this case, we need to use numerical methods to approximate the posterior distribution. One of the simplest methods is the grid approximation.

The grid approximation is to approximate the posterior distribution by a discrete distribution. For instance, we can approximate the posterior distribution by a discrete distribution with $K$ points $\theta_1, \theta_2, \dots, \theta_K$. For each $\theta_i$, we can compute the posterior probability $p(\theta_i | Y = 9)$. Then, we can approximate the posterior distribution by the discrete distribution. 

The following code gives the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on beta-binomial bayesian model and grid approximation.

```R
# Define a grid 
grid_data <- data.frame(theta_grid = seq(from = 0, to = 1, by = 0.01))

grid_data %>%
    # calculate the prior probability and likelihood
    mutate(piror = dbeta(theta_grid, 2, 2),
                likelihood = dbinom(9, 10, theta_grid)) %>%
    # calculate the posterior probability
    mutate(unnormalized = piror * likelihood,
            posterior = unnormalized / sum(unnormalized)) %>%
    with(plot(x = theta_grid, y = posterior, type = "h", lwd = 2,
                xlab = "Theta", ylab = "Posterior Probability",
                main = "Posterior of Theta"))
```

The above code shows that instead of using the beta distribution, we can use the discrete distribution with many points to approximate the posterior distribution by using Bayesian updating formula.

<div class='figure'>
    <img src="/math/images/posterior_grid_approx1.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The plot of the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on grid approximation.
    </div>
</div>

We can compare the posterior distribution of $\theta$ in Figure 1 with the beta distribution in Figure 2. We can see that the posterior distribution of $\theta$ in Figure 1 is very close to the beta distribution in Figure 2.


<div class='figure'>
    <img src="/math/images/beta_grid_approx2.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 2.</span> The plot of the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on beta distribution.
    </div>
</div>

If you look at Figure 1 and 2 carefully, you will find that the y-axis in Figure 1 is different from the y-axis in Figure 2. The y-axis in Figure 1 is the posterior probability, while the y-axis in Figure 2 is the density of the beta distribution. To create a density plot, we have to sample from the posterior distribution and then use the density function to calculate the density. The following code shows how to create a density plot of the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on grid approximation.

{% endkatexmm %}

```R
# sample from grid_data
set.seed(89756)

grid_data %>%
    # calculate the prior probability and likelihood
    mutate(piror = dbeta(theta_grid, 2, 2),
                likelihood = dbinom(9, 10, theta_grid)) %>%
    # calculate the posterior probability
    mutate(unnormalized = piror * likelihood,
            posterior = unnormalized / sum(unnormalized)) %>%
    sample_n(10000, weight = posterior, replace = TRUE) -> grid_sample


grid_sample %>%
    with(hist(x = theta_grid, prob = TRUE, main = "Posterior of Theta",
            xlab = "Theta", ylab = "Probability Density", lwd = 2)) %>%
    with(lines(density(grid_sample$theta_grid),
                          col = "blue", lwd = 2)) %>%
    with(curve(dbeta(x, 11, 3), from = 0, to = 1, lwd = 2,
                            col = "red", add=TRUE)) %>%
    with(legend(0.24, 3.8,
                    legend = c("Posterior", "Beta", "Estimated Density"),
                    col = c("gray", "red", "blue"),
                    lty = 1, lwd = 2, y.intersp = 1.5))
```

{% katexmm %}

<div class='figure'>
    <img src="/math/images/grid_approx_sample.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 3.</span> The plot of the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on beta distribution and grid approximation.
    </div>
</div>


## MCMC approximation with rstan

We will continue to use the same model as above. However, we will use MCMC to approximate the posterior distribution with rstan. Stan is a probabilistic programming language that is used to fit Bayesian models. The following code shows how to use rstan to approximate the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$.

There are three key elements in Stan: data, parameters, and model. The data element is used to store the observed data. The parameters element is used to store the parameters that we want to estimate. The model element is used to store the model. The following code shows how to define the data, parameters, and model in Stan.

```R
bb_model <- "
    data {
        int<lower=0, upper=10> Y;
    }
    parameters {
        real<lower=0, upper=1> theta;
    }
    model {
        theta ~ beta(2, 2);
        Y ~ binomial(10, theta);
    }
"

# compile the model
bb_model_sim <- stan(model_code = bb_model,
                                    data = list(Y = 9),
                                    chains = 4,
                                    iter = 10000,
                                    seed = 89756)

bb_model_sim %>% 
    as.array() %>% dim()  # 4 chains, 5000 iterations
# 5000, 4, 2

bb_model_sim %>%
    as.array() -> bb_model_sim_array

bb_model_sim_array[1:5, 1:4, 1] %>%
        kable("pipe", digits = 3)
```

The above code gives the following table, which shows the first five iterations of the four chains. Each chain is a Markov chain. Instead of searching in the grid space, the Markov chain searches in the parameter space. The Markov chain starts from a random point in the parameter space and then moves to a new point in the parameter space. The new point is chosen based on the current point. We will discuss the algorithm of the Markov chain in the next post.


| chain1| chain2| chain3| chain4|
|:-------:|:-------:|:-------:|:-------:|
|   0.791|   0.743|   0.736|   0.852|
|   0.803|   0.844|   0.844|   0.759|
|   0.777|   0.901|   0.878|   0.822|
|   0.973|   0.910|   0.896|   0.877|
|   0.917|   0.882|   0.783|   0.769|



<div class='figure'>
    <img src="/math/images/approx_stan_trace1.png"
         alt="Inequality bounds compare"
         style="width: 70%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 4.</span> The trace plot of the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on MCMC approximation.
    </div>
</div>

Marking the sequence of the chain values, the trace plots in Figure 4 illuminate the Markov chains’ longitudinal behavior. We also want to examine the distribution of the values these chains visit along their journey, ignoring the order of these visits. Figure 5 gives the distribution of posterior, which shows that the estimation is very close to the true value (beta distribution).

<div class='figure'>
    <img src="/math/images/rstan_approx_posterior.png"
         alt="Inequality bounds compare"
         style="width: 80%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 5.</span> The hist and density plot of the posterior distribution of $\theta$ when $Y = 9$ and $n = 10$ based on MCMC approximation.
    </div>
</div>






{% endkatexmm %}











