library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, janitor, readxl, png, Cairo, rbenchmark,
    httr, jsonlite, fedmatch, patchwork,
    corrplot, tidygraph, ggraph, igraph,
    treemap, splitstackshape, stringr, lubridate,
    poweRlaw, voronoiTreemap, ggridges,
    DescTools, stringi, kit, pheatmap, rstan
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
                    legend = c("Posterior", "Beta", "Estimated Density"),
                    col = c("gray", "red", "blue"),
                    lty = 1, lwd = 2, y.intersp = 1.5))
    

