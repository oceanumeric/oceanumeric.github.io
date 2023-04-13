library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, janitor, readxl, png, Cairo, rbenchmark,
    httr, jsonlite, fedmatch, patchwork,
    corrplot, tidygraph, ggraph, igraph,
    treemap, splitstackshape, stringr, lubridate,
    poweRlaw, voronoiTreemap, ggridges,
    DescTools, stringi, kit, pheatmap, rstan, bayesplot
)
# set option
options(dplyr.summarise.inform = FALSE)
options(jupyter.plot_mimetypes = "image/png")
options(repr.plot.width = 8, repr.plot.height = 5)

gray_scale <- c("#F3F4F8", "#D2D4DA",  "#bcc0ca",
                "#D3D3D3", "#B3B5BD", "#838383",
                "#9496A1", "#7d7f89", "#777986",
                "#656673", "#5B5D6B", "#4d505e",
                "#404352", "#2b2d3b", "#2B2B2B", "#282A3A",
                "#1b1c2a", "#191a2b", "#141626", "#101223")
color_scale <- c("#BB4444", "#EE9988", "#FFFFFF",
                            "#77AADD", "#4477AA")
blue_scale <- c("#DBE7F5", "#95BCE3",
                "#699BCB", "#6597C7", "#4879A9",
                "#206BAD", "#11549B")

cor_col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF",
                            "#77AADD", "#4477AA"))

# check working directory
getwd()

peepsample <- function(dt) {
    dt %>%
    .[sample(.N, 5)] %>%
    kable("pipe")
}

peephead <- function(dt, x=5) {
    dt %>%
    head(x) %>%
    kable("pipe")
}

### ------- grid approximation ------- ###
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


curve(dbeta(x, 11, 3), from = 0, to = 1, lwd = 2,
        main = "Posterior of Theta: Beta(11, 3)", xlab = "Theta",
        ylab = "Posterior Probability")

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
                    legend = c("Posterior", "Beta(11, 3)", "Estimated Density"),
                    col = c("gray", "red", "blue"),
                    lty = 1, lwd = 2, y.intersp = 1.5))
    

### ------- MCMC with rstan ------- ###

# define the model
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

# plot for one chain
bb_model_sim_array[, 1:4, 1] %>%
    as.data.table() %>%
    setnames(c("chain1", "chain2", "chain3", "chain4")) %>%
    # plot chain4
    with(plot(chain4, type = 'l',
                col = gray_scale[7],
                xlab = "Iteration", ylab = "Theta",
                main = "Trace Plot of Theta for Chain 4"))


# plot all chains
mcmc_trace(bb_model_sim, pars = "theta", size = 0.1)

# plot hist and density

bb_model_sim %>%
    as.data.table() -> bb_model_dt

options(repr.plot.width = 9, repr.plot.height = 5)
par(mfrow = c(1, 2))
bb_model_dt %>%
    with(hist(theta, main = "Posterior of Theta",
            xlab = "Theta", ylab = "Count",
            lwd = 2))
plot(density(bb_model_dt$theta), col = "blue", lwd = 1.5,
                xlab = "Theta", ylab = "Probability Density",
                main = "Posterior of Theta")
curve(dbeta(x, 11, 3), from = 0, to = 1, lwd = 1.5, lty = 2,
                        col = "red", add = TRUE)
legend("topleft", legend = c("Estimation", "Beta(11, 3)"),
        bg = "transparent", box.col = "transparent",
        col = c("blue", "red"),
        lty = c(1, 2), lwd = c(1.5, 1.5))
    

### ------- MCMC with rstan: Gamma Poisson model ------- ###

# sequence data y1, y2 as number of events in each time interval
# y is integer 
gp_model <- "
    data {
        int<lower=0> Y[2];
    }
    parameters {
        real<lower=0> lambda;
    }
    model {
        lambda ~ gamma(3, 1);
        Y ~ poisson(lambda);
    }
"

# run the simulation
gp_sim <- stan(model_code = gp_model,
                            data = list(Y = c(2, 8)),
                            chains = 4,
                            iter = 10000,
                            seed = 89756)

# plot the trace plot, histogram and density
mcmc_trace(gp_sim, pars = "lambda", size = 0.1) -> p1

mcmc_hist(gp_sim, pars = "lambda") + yaxis_text(TRUE) + ylab("Count") -> p2

mcmc_dens(gp_sim, pars = "lambda") + yaxis_text(TRUE) + 
                            ylab("Probability Density") -> p3

options(repr.plot.width = 9, repr.plot.height = 6)
p1 / (p2 + p3)


# compare parallel chains
options(repr.plot.width = 8, repr.plot.height = 5)
mcmc_dens_overlay(gp_sim, pars = "lambda") + ylab("Probability Density") 

# simulate a short model
gp_sim_short <- stan(model_code = gp_model,
                            data = list(Y = c(2, 8)),
                            chains = 4,
                            iter = 100,
                            seed = 89756)

# compare the two models
mcmc_trace(gp_sim_short, pars = "lambda")
mcmc_dens_overlay(gp_sim_short, pars = "lambda")

neff_ratio(gp_sim)

options(repr.plot.width = 7, repr.plot.height = 5)
mcmc_acf(gp_sim, pars = "lambda")

rhat(gp_sim)