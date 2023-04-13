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

### --- Simulate Metropolis-Hastings Algorithm --- ###
# scenario 2
y_data <- 6.25
current_u <- 3

set.seed(8)
# set width = 1 and draw from uniform distribution
proposal_u <- runif(1, min = current_u - 1, max = current_u + 1)
# proposal_u = 2.93259

# calculate likelihood
# likelihood is just the probability of the data given the parameter
likelihood_current <- dnorm(y_data, mean = current_u, sd = 0.75)
likelihood_proposal <- dnorm(y_data, mean = proposal_u, sd = 0.75)
# calculate bayesian product = likelihood * prior
bayes_prod_current <- likelihood_current * dnorm(current_u, mean = 0, sd = 1)
bayes_prod_proposal <- likelihood_proposal * dnorm(proposal_u, mean = 0, sd = 1)

# calculate alpha
alpha <- min(1, bayes_prod_proposal / bayes_prod_current)

# accept or reject
next_u <- sample(c(proposal_u, current_u), size = 1,
                        prob = c(alpha, 1 - alpha))


# make a table to show the results
mh_table <- data.frame(
    current_u = current_u,
    proposal_u = proposal_u,
    likelihood_current = likelihood_current,
    likelihood_proposal = likelihood_proposal,
    bayes_prod_current = bayes_prod_current,
    bayes_prod_proposal = bayes_prod_proposal,
    alpha = alpha,
    next_u = next_u
)

# transpose the table
mh_table %>%
    t() %>%
    kable("pipe")


## write a function to simulate the MH algorithm
one_mh_iteration <- function(w, current, y_data) {
    # a function to simulate one iteration of the MH algorithm
    # w is the width of the proposal distribution
    # current is the current value of the parameter
    # we are using unif(w) to draw from the proposal distribution
    # prior is normal(0, 1)
    # likelihood is normal(y_data, sd = 0.75)

    # draw from the proposal distribution
    proposal <- runif(1, min = current - w, max = current + w)

    # update the parameter based on the bayesian product
    # calculate the likelihood
    likelihood_current <- dnorm(y_data, mean = current, sd = 0.75)
    likelihood_proposal <- dnorm(y_data, mean = proposal, sd = 0.75)

    # calculate the bayesian product
    bayes_prod_current <- likelihood_current * dnorm(current,
                                                            mean = 0,
                                                            sd = 1)
    bayes_prod_proposal <- likelihood_proposal * dnorm(proposal,
                                                            mean = 0,
                                                            sd = 1)

    # calculate alpha
    alpha <- min(1, bayes_prod_proposal / bayes_prod_current)

    # accept or reject
    next_u <- sample(c(proposal, current), size = 1,
                        prob = c(alpha, 1 - alpha))

    return(data.frame(proposal, alpha, next_u))


}

set.seed(8)
one_mh_iteration(1, 3, 6.25)

set.seed(83)
one_mh_iteration(1, 3, 6.25)

seed_vec <- c(8, 83, 7) # set seed
foo <- data.frame()
for (s in seed_vec) {
    set.seed(s)
    temp <- one_mh_iteration(1, 3, 6.25)
    foo <- rbind(foo, temp)
}

foo$seed <- seed_vec

# reorder the columns
foo[, c("seed", "proposal", "alpha", "next_u")] %>%
    kable("pipe", digits = 3, align = 'l')