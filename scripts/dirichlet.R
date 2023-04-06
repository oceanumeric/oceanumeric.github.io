# plot binomial distribution
binomial_fun <- function(n, p, k) {
  choose(n, k) * p^k * (1-p)^(n-k)
}

kvec <-  0:40

p1 <- binomial_fun(20, 0.5, kvec)
p2 <- binomial_fun(20, 0.75, kvec)
p3 <- binomial_fun(40, 0.75, kvec)

# set width
options(repr.plot.width = 7, repr.plot.height = 5)

# save the plot
png("binomial.png", width = 7, height = 5, units = "in", res = 300)
plot(kvec, p1, type = "b", ylim = c(0, 0.27),
            xlab = "k", ylab = "p(k)",
            main = "Binomial Distribution")
points(kvec, p2, pch = 4)
lines(kvec, p2, lty = "dashed")
points(kvec, p3, pch = "+")
lines(kvec, p3, lty = "dotted")

legend("topright", legend = c("n = 20, p = 0.5",
                              "n = 20, p = 0.75",
                              "n = 40, p = 0.75"),
       lty = c(1, 2, 3), pch = c(1, 4, 3))
dev.off()

# plot Poisson distribution
poisson_fun <- function(lambda, k) {
  lambda^k * exp(-lambda) / factorial(k)
}

kvec <-  0:20
p1 <- poisson_fun(1, kvec)
p2 <- poisson_fun(4, kvec)
p3 <- poisson_fun(10, kvec)

# save the plot
png("poisson_distribution.png", width = 7, height = 5, units = "in", res = 300)
plot(kvec, p1, type = "b", ylim = c(0, 0.4),
            xlab = "k", ylab = "p(k)",
            main = "Poisson Distribution")
points(kvec, p2, pch = 4)
lines(kvec, p2, lty = "dashed")
points(kvec, p3, pch = "+")
lines(kvec, p3, lty = "dotted")
legend("topright", legend = c("lambda = 1",
                              "lambda = 4",
                              "lambda = 10"),
       lty = c(1, 2, 3), pch = c(1, 4, 3))
dev.off()