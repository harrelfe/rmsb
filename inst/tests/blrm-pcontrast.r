require(rmsb)
options(mc.cores=parallel::detectCores() - 1,
        rmsb.backend='cmdstan', rmsbdir='~/.rmsb')
cmdstanr::set_cmdstan_path('/usr/local/bin/cmdstan-2.32.2')
set.seed(1)
x <- rnorm(100)
x2 <- sample(c('a', 'b', 'c'), 100, TRUE)
y <- round(x + 0.4 * x^2 + (x2 == 'b') + 1.5 * (x2 == 'c') + 0.5 * rnorm(100), 1)
dd <- datadist(x, x2); options(datadist='dd')
ggplot(Predict(orm(y ~ pol(x,2) + x2), x, x2))
f <- blrm(y ~ pol(x, 2) + x2)
ggplot(Predict(f, x, x2))

# Define an abbreviation for list()
. <- function(...) list(...)

# Define a function to create the list needed by pcontrast since
# we want to vary sd but keep everything else the same
pcon <- function(sd) list(sd=sd, c1=.(x=-1), c2=.(x=0), c3=.(x=1),
                          contrast=expression(0.5 * (c1 + c3) - c2))

g <- blrm(y ~ pol(x, 2) + x2, keepsep='x', pcontrast=pcon(3))
h <- blrm(y ~ pol(x, 2) + x2, keepsep='x', pcontrast=pcon(0.3))

# Force litle nonlinearity AND tiny difference between x2=b and x2=c
pcon <- function(sd) list(sd=sd, c1=.(x=-1), c2=.(x=0), c3=.(x=1),
                          c4=.(x2='b'), c5=.(x2='c'),
                          contrast=expression(0.5 * (c1 + c3) - c2,
                                              c4 - c5) )
i <- blrm(y ~ pol(x, 2) + x2, keepsep='x', pcontrast=pcon(c(0.1, 0.1)))
i$Contrast

# lapply(list(g, h, i), stanDx)

b <- grep('x', names(coef(f)))
w <- lapply(llist(f, g, h, i), function(x) coef(x)[b])
do.call(rbind, w)
ggplot(Predict(i, x, x2))
