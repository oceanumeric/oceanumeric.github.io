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
