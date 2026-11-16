require(rmsb)
options(prType='html')
require(ordinal)
cstanSet()
set.seed(2)
n <- 500
x <- c(rep(0, 250), rep(1, 250))
y <- c(rnorm(250, 0, 1), rnorm(250, 0, 3))
y <- round(2 * y) / 2
# y <- round(y, 1)
length(unique(y))
r <- max(abs(range(y)))
histbackback(split(y, x), brks=seq(-r, r, by=1))
f <- borm(y ~ x, family='probit')
f <- blrm(y ~ x, ~ x, cppo = function(y) y)
f <- clm(as.ordered(y) ~ x, scale = ~ x, link = "probit")
summary(f)
sopts <- list(adapt_delta=0.995, max_treedepth=15)
sopts <- NULL
f <- borm(y ~ x, yde = ~ x, ytrans = function(y) y, family='probit', sampling.args=sopts)
f <- borm(y ~ x, yde = ~ x, ytrans = function(y) cbind(y=y, y2=y^2), family='probit', sampling.args=sopts)
f
