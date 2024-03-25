# Compare intercepts and betas with MLE

require(rmsb)
set.seed(1)
y <- rep(1:10, c(2, 6, 3, 12, 4, 16, 5, 24, 6, 34))
n <- length(y)
x <- ifelse(runif(n) < 0.4, y, sample(1:10, n, TRUE))
f <- lrm(y ~ x)
g0 <- blrm(y ~ x, method='optimizing', iprior=0)
g1 <- blrm(y ~ x, method='optimizing', iprior=1)
g2 <- blrm(y ~ x, method='optimizing', iprior=2)
cbind(coef(f), coef(g0), coef(g1), coef(g2))
