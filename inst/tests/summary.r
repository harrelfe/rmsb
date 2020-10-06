require(rmsb)
d <- data.frame(HE6=as.integer(cut2(sample(1:100, 200, replace = TRUE), g=10)),
                  Age = sample(1:85, 200, replace = TRUE), EORTC = sample(1:100, 200, replace = TRUE),
                  linf=rbinom(200, 1,.5),
                  cir=rbinom(200, 1,.5))

head(d)

dd <- datadist(d)
options(datadist='dd')

g <- blrm(HE6  ~ cir*rcs(Age,3) + pol(EORTC) + linf,
               ~ cir*rcs(Age,3) + pol(EORTC),
          cppo = function(y) y, data=d)
s <- summary(g, ycut=5)
s
plot(s)
exprob <- ExProb(g)

fun5 <-function(x) exprob (x, y=5)
fun6 <-function(x) exprob (x, y=6)
fun9 <-function(x) exprob (x, y=9)

M   <- Mean(g)
qu  <- Quantile(g)
med <- function (lp) qu(.5 , lp)

# nomogram will not work with partial PO model; use the ols trick
# Get linear predictor for Prob y >= 5
xb5 <- predict(g, d, ycut=5, cint=0)
h5 <- ols(xb5 ~ cir*rcs(Age,3) + pol(EORTC) + linf, data=d)
h5$stats['R2']   # 1.0;  if < 1.0 there is an inconsistency between two formulas for blrm
# Get a second linear predictor for y >= 9 and see if this is a
# constant away from the first one
xb9 <- predict(g, d, ycut=9, cint=0)
range(xb9 - xb5)  # no, difference is covariate-specific so this approach won't work

# The following is probably riht for Prob y>=5; not sure if mean is correct
n <- nomogram(h, fun=list(Mean=M, "Prob y>=5"=fun5))
plot(n)
