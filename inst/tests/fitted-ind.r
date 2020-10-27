## See https://github.com/harrelfe/rmsb/issues/4
require(rmsb)
set.seed(836)
data<- data.frame(HE6=sample(1:10, 200, replace = TRUE,
                             prob=c(rep(0.1,6),0.01,0.002,0.19,0.198) ), 
                  Age = sample(1:85, 200, replace = TRUE),
                  EORTC = sample(1:100, 200, replace = TRUE), 
                  linf=rbinom(200, 1,.5),
                  cir=rbinom(200, 1,.5),
                  esquema=rbinom(200, 1,.5),
                  riesgo=factor(rbinom(200, 2,.5)),
                  estadio=factor(rbinom(200, 2,.5)))
head(data)
table(data$HE6)
dd <- datadist(data)
options(datadist='dd')

newdata <- data.frame(cir=0, Age=85, EORTC= 10, linf=0, riesgo=0, esquema=1,
                      estadio=1)

## Following gave some negative lower limits but not point estimates for probs
f <- blrm(HE6 ~ cir*rcs(Age, 3) + linf + pol(EORTC), ~ rcs(Age, 3) + pol(EORTC),
          cppo=function(y) y, data=data)

## No negative point estimates for seed=1,...,10
for(s in 4 : 10) {
  cat('s:', s, '\n')
f <- blrm(HE6  ~ cir*rcs(Age,3)+ linf+ pol(EORTC)+esquema+estadio+riesgo, 
            ~ rcs(Age,3)+ pol(EORTC), cppo=function(y) y, data=data, seed=s) 

  print(predict(f, newdata, type='fitted.ind'))
}

