require(rmsb)
stanSet()
set.seed(11)
n <- 200000
x <- rnorm(n)
y <- as.integer(cut2(x + rnorm(n), g=6))
f <- orm(y ~ x)
coef(f)
cp <- function(y) y
g <- blrm(y ~ x, ~ x, cppo=cp, method='optimizing')
coef(g)
a <- b <- y

# Left censor 1/3 of obs with y <= 2
r <- runif(n) < 1/3
i <- y <= 2 & r
a[i] <- 1
b[i] <- 2
# Interval censor obs with y = 3 or 4
i <- (y == 3 | y == 4) & r
a[i] <- 3
b[i] <- 4
# Right censor obs with y = 5 or 6
i <- y >= 5 & r
a[i] <- 5
b[i] <- 6
table(y, paste0('[', a, ',', b, ']'))
Y <- Ocens(a, b)
h <- blrm(Y ~ x, ~x, cppo=cp)   # in ~/tmp/h.rds
res <-
  rbind('No cens:mode'=coef(g),
        'Cens:mode'   =coef(h, 'mode'),
        'Cens:mean'   =coef(h, 'mean'))
res

#                  y>=2     y>=3         y>=4      y>=5      y>=6        x     x x f(y)
# No cens:mode 2.338427 1.025784 -0.002575473 -1.029774 -2.338639 1.701404 -0.005007498
# Cens:mode    2.394846 1.053724  0.016988116 -1.008903 -2.285029 1.720565 -0.066613448
# Cens:mean    2.394922 1.053770  0.016974146 -1.008947 -2.285118 1.720541 -0.066633208

