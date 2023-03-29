library(dplyr)
library(data.table)
library(knitr)
# plot beta distribution for different values of alpha and beta
# save the plot as beta_distribution.png

png("./work/docs/math/images/beta_distribution.png", width = 6, height = 5,
                units = "in", res = 300)
p = seq(0, 1, length.out = 100)
alpha = c(0.5, 1, 2, 3)
beta = c(0.5, 1, 2, 3)
plot(p, dbeta(p, alpha[1], beta[1]), type = "l", col = "red", lwd = 2, 
    xlab = "p", ylab = "density", main = "Beta Distribution")
for (i in 2:length(alpha)) {
    lines(p, dbeta(p, alpha[i], beta[i]), col = i, lwd = 2)
}
legend("topright", legend = paste("alpha =", alpha, "beta =", beta), 
    col = 1:length(alpha), lty = 1, lwd = 2)
dev.off()


# find values of alpha and beta that satify the following conditions:
# E[p] = 0.5
# P(p < 0.01) = 0.01

# prior probability of success
# the value can be changed based on the prior knowledge
# in practice, we set up alpha and beta and search for 
# the value of p that satisfies the conditions
prior_p = 100/1000 
prior_mean = 500/1000

# generate a grid of alpha and beta
alpha = seq(0.001, 10, length.out = 1000)
beta = alpha * (1 - prior_mean) / prior_mean

# calcuate the prior probability based on beta distribution
prior = pbeta(prior_p, alpha, beta)

# plot the prior probability for alpha and beta in two plots and
# combine them into one plot

png("../math/images/beta_prior.png", width = 8, height = 4,
                units = "in", res = 300)
par(mfrow = c(1, 2))
plot(alpha, prior, type = "l", xlab = "alpha",
            ylab = "prior probability", 
            main = "Prior Probability - alpha")
abline(h=0.01, col="red", lty=2)

plot(beta, prior, type = "l", xlab = "beta",
            ylab = "prior probability", 
            main = "Prior Probability - beta")
abline(h=0.01, col="red", lty=2)
dev.off()

# get alpha and beta that satisfy the conditions
# calculate the probability that is close to 0.01
p_dist = abs(prior-0.01)
idx = which(p_dist == min(p_dist))
alpha[idx]  # 2.863
beta[idx]  # 2.863

# plot the beta distribution for the values of alpha and beta
png("../math/images/beta_prior2.png", width = 6, height = 4,
                units = "in", res = 300)
p_seq = seq(0, 1, length.out = 1000)
plot(p_seq, dbeta(p_seq, alpha[idx], beta[idx]), type = "l", 
    xlab = "p", ylab = "density", 
    main = "Beta Distribution - alpha = 2.863, beta = 2.863")
abline(v = prior_p, col = "red", lty = 2)
abline(v = prior_mean, col="gray", lty=2)
dev.off()


# calculate the posterior probability of success
x = 3
n = 10

# prior probability of success for three categories
# good player, average player, bad player
binom_p = c(0.7, 0.5, 0.1)
prior = c(0.2, 0.75, 0.05)
# calculate the likelihood of success
likelihood = dbinom(x, n, binom_p)

# 0.0090016920.11718750.057395628
# each category has a different likelihood of success

# calculate the posterior probability of success
posterior = likelihood * prior / sum(likelihood * prior)

# create a data frame to store the results
df = data.frame(binom_p, prior, likelihood, posterior)
names(df) = c("binom_p", "prior", "likelihood", "posterior")

df %>%
    round(3) %>%
    kable("markdown", align = "c")


# posterior probability based on continuous beta distribution
# prior probability of success
png("../math/images/beta_prior3.png", width = 6, height = 4,
                units = "in", res = 300)
x = 3
n = 10
p_seq = seq(0, 1, length.out = 1000)
plot(p_seq, dbeta(p_seq, alpha[idx], beta[idx]), type = "l", 
    xlab = "p", ylab = "density", ylim = c(0, 4),
    main = "Update Beta Distribution")
lines(p_seq, dbeta(p_seq, alpha[idx] + x, beta[idx] + n - x), 
    col = "red", lwd = 2)
legend("topright", legend = c("prior", "posterior"), 
    col = c("black", "red"), lty = 1, lwd = 2)
dev.off()