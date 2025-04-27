# Compare intercepts and betas with MLE
# First try discrete case with lots of ties and alternating low
# and high frequencies in y

require(rmsb)
cstanSet()
set.seed(1)
y <- rep(1:10, c(2, 6, 3, 12, 4, 16, 5, 24, 6, 34))
n <- length(y)
x <- ifelse(runif(n) < 0.4, y, sample(1:10, n, TRUE))
d <- data.frame(x, y)

test <- function() {
  f <- lrm(y ~ x, data=d)
  g0 <- blrm(y ~ x, method='optimizing', iprior=0, data=d)
  g1 <- blrm(y ~ x, method='optimizing', iprior=1, data=d)
  g2 <- blrm(y ~ x, method='optimizing', iprior=2, ascale=20, data=d)
  k <- coef(f); k0=coef(g0); k1=coef(g1); k2=coef(g2)
  print(cbind(k, k0, k1, k2))
  u = cbind(k0 - k, k1 - k, k2 - k)
  apply(u, 2, function(x) max(abs(x)))
}

test()
# 0.0002 0.0007 0.008
# Winner: dirichlet
# ascale much < 20 gave more disagreement with mle

# Now try continuous case with no ties
y <- 1:200
x <- ifelse(runif(200) < 0.4, y, sample(1:200, 200, TRUE))
d <- data.frame(x, y)
test()
# Multiple runs
# 0.0002   0.001   0.086
# 0.00007  0.0006  0.087
# 0.0002   0.001   0.086
# 0.0001   0.004   0.086
# 0.0005   0.0003  0.092
# 0.0002   0.001   0.084

# Winner: dirichlet

