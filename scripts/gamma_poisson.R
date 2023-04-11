library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, janitor, readxl, png, Cairo, rbenchmark,
    httr, jsonlite, fedmatch, patchwork,
    corrplot, tidygraph, ggraph, igraph,
    treemap, splitstackshape, stringr, lubridate,
    poweRlaw, voronoiTreemap, ggridges,
    DescTools, stringi, kit, pheatmap
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

### ------- plot poisson pmf ------ ###
x_seq <- seq(0, 12, 1)
lambda_seq <- c(1, 2, 5)


plot_poisson <- function(lambda) {
    y_seq <- dpois(x_seq, lambda = lambda)
    main_str = paste("Poisson(", lambda, ") PMF", sep = "")
    plot(x_seq, y_seq, type = "h", lwd = 2,
        xlab = "Number of events (k)", ylab = "P(X=k)",
        ylim = c(0, 0.5),
        main = main_str)
    points(x_seq, y_seq, pch = 16, cex = 1)
}

options(repr.plot.width = 8, repr.plot.height = 3.5)
par(mfrow = c(1, 3))
for (lambda in lambda_seq) {
    plot_poisson(lambda)
}


### ------- plot exponential distribution ----- ###

curve(dexp(x, rate = 1), from = 0, to = 10, lwd = 1.5,
    col = "blue",
    xlab = "Time (t)", ylab = "P(X=t)",
    main = "Exponential PDF")
curve(dexp(x, rate = 2), from = 0, to = 10, lwd = 1.5,
    col = "red", add = TRUE)
curve(dexp(x, rate = 5), from = 0, to = 10, lwd = 1.5,
    col = "purple", add = TRUE)
legend(7, 0.9, legend = c("rate = 1", "rate = 2", "rate = 5"),
    col = c("blue", "red", "purple"), lwd = 1.5, cex = 1.2)
