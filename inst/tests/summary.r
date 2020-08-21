require(rmsb)
d <- data.frame(HE6=as.integer(cut2(sample(1:100, 200, replace = TRUE),g=10)),
                Age = sample(1:85, 200, replace = TRUE), EORTC = sample(1:100, 200, replace = TRUE),
                cir=rbinom(200, 1,.5))
head(d)

dd <- datadist(d)
options(datadist='dd')

g <- blrm(HE6  ~ cir*rcs(Age,3)+ pol(EORTC),
               ~ cir*rcs(Age,3)+ pol(EORTC), cppo=function(y) y, data=d)
summary(g, ycut=5)

f <- blrm(HE6  ~ cir,
            ~ cir, cppo=function(y) y, data=d)
summary(f, ycut=5)
plot(summary(f, ycut=5))

newdata <- data.frame(cir=c(0,1), Age=53, EORTC= 75)
predict(f, newdata, type='fitted') #

ggplot(Predict(f), abbrev =TRUE , ylab=NULL)

exprob <- ExProb(f)

fun5<-function(x) exprob (x, y=5)
fun6<-function(x) exprob (x, y=6)
fun9<-function(x) exprob (x, y=9)

M <- Mean(f)
qu <-Quantile(f)
med <- function (lp, ...) qu(.5 , lp, ...)

n <- nomogram(f, fun=list(Mean=M, "Prob y>=5"=fun5,"Prob y >=6"=fun6, "Prob y>=9"=fun9), lp=FALSE)
plot(n)
