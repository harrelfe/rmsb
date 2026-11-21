require(rmsb)
# options(prType='html')
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
sopts <- list(adapt_delta=0.995, max_treedepth=15, seed=7)
# sopts <- NULL
fi <- function(y, bar) y - bar
formals(fi)$bar <- mean(y)
f1 <- orm(y ~ x, ~ x, ytrans=fi, family='probit')
f1
f2 <- borm(y ~ x, ~ x, fi, family='probit', sampling.args=sopts)
f2
g <- function(y, bar) cbind(lin=y - bar, quad=(y - bar) ^ 2)
formals(g)$bar <- mean(y)
# The following borm call took 694s
# f <- borm(y ~ x, yde = ~ x, ytrans = g, family='probit', sampling.args=sopts)
f <- orm(y ~ x, yde = ~ x, ytrans = g, family='probit')
f
