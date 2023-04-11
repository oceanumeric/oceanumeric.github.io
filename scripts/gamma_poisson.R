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


### ------- plot gamma distribution ----- ### 
plotGamma <- function(shape=2, rate=0.5, to=0.99, p=c(0.1, 0.9), cex=1, ...){
  to <- qgamma(p=to, shape=shape, rate=rate)
  curve(dgamma(x, shape, rate), from=0, to=to, n=500, type="l", 
        main=sprintf("gamma(x, shape=%1.2f, rate=%1.2f)", shape, rate),
        bty="n", xaxs="i", yaxs="i", xlab="", ylab="", 
        las=1, lwd=1.5, cex=cex, cex.axis=cex, cex.main=cex, ...)
  gx <- qgamma(p=p,  shape=shape, rate=rate)
  gy <- dgamma(x=gx, shape=shape, rate=rate)
  for(i in seq_along(p)) { lines(x=rep(gx[i], 2), y=c(0, gy[i]), col="blue") }
  for(i in seq_along(p)) { text(x=gx[i], 0, p[i], adj=c(1.1, -0.2), cex=cex) }
}

options(repr.plot.width = 9, repr.plot.height = 6)
par(mfrow=c(2, 3))
plotGamma(1, 0.1)
plotGamma(2, 0.1)
plotGamma(6, 0.1)
plotGamma(1, 1)
plotGamma(2, 1)
plotGamma(6, 1)


### ------- Gamma-Poisson conjugate ----- ###
n <- 5
lambda_true <- 3
set.seed(123)
y <- rpois(n, lambda_true)


# chose prior
alpha <- 1
beta <- 1

# set lambda sequence
lambda_seq <- seq(0, 7, 0.1)

options(repr.plot.width = 7, repr.plot.height = 5)
plot(lambda_seq, dgamma(lambda_seq, shape = alpha, rate = beta),
    type = "l", lwd = 2, 
    xlab = "lambda", ylab = "P(lambda)",
    main = "Gamma Prior")
lines(lambda_seq, dgamma(lambda_seq, shape = alpha + sum(y), rate = beta +n),
    type = "l", lwd = 2, col = "blue")
abline(v = lambda_true, lty = 2)
legend('topright', inset = .02, legend = c('prior', 'posterior'),
       col = c('black', 'blue'), lwd = 2)


# plot the posterior distributions with different 
# sample sizes to see if things even out:

n_total <- 200
set.seed(111111) # use same seed, so first 5 obs. stay same
y_vec <- rpois(n_total, lambda_true)
lambda <- seq(0, 7, 0.1)

n_vec <- c(1, 2, 5, 10, 50, 100, 200)

options(repr.plot.width = 9, repr.plot.height = 12)
par(mfrow = c(4,2), mar = c(2, 2, .1, .1))
plot(lambda, dgamma(lambda, alpha, beta), type = 'l', lwd = 2, col = 'orange',
     ylim = c(0, 3.2), xlab = '', ylab = '')
abline(v = lambda_true, lty = 2)
text(x = 0.5, y = 2.5, 'prior', cex = 1.75)

for(n_crnt in n_vec) {
  y_sum <- sum(y_vec[1:n_crnt])
  plot(lambda, dgamma(lambda, alpha, beta), type = 'l', lwd = 2, col = 'orange',
       ylim = c(0, 3.2), xlab = '', ylab = '')
  lines(lambda, dgamma(lambda, alpha + y_sum, beta + n_crnt), 
        type = 'l', lwd = 2, col = 'violet')
  abline(v = lambda_true, lty = 2)
  text(x = 0.5, y = 2.5, paste0('n=', n_crnt), cex = 1.75)
}