require(rmsb)
n <- 500   # subjects
set.seed(2)
re <- rnorm(n) * 4
X <- runif(n)   # baseline covariate, will be duplicated over repeats
m <- 10         # measurements per subject

id <- rep(1 : n, each = m)
x  <- X[id]
L <- x + re[id]   # actual logit
y <- round((L + 2 * rnorm(length(L))) / 5)
f <- blrm(y ~ x + cluster(id), iprior=2, ascale=2.5)
## Sampling not very good although sigma gamma is fine
## With default iprior=0 or with iprior=1 sampling was not very good either

f <- blrm(y ~ x)
## Sampling fine

y2 <- 1 * (y >= 1)
f <- blrm(y2 ~ x + cluster(id))
## Sampling fair  n_eff for Intercept 549
