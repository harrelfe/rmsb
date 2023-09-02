## From Christopher Selman
## https://discourse.datamethods.org/t/bayesian-regression-modeling-strategies/6105/78

## Very simple simulation with one binary variable (eg two-arm trial) and an
# ordinal outcome with 5 categories
require(rmsb)
options(mc.cores = parallel::detectCores())
set.seed(02082023)

# Treatment probs
p0 <- c(rep(1,5))
p1 <- c(rep(1,2),rep(2,3))

states <- c("A", "B", "C","D","E")

# Simulate two arm trial data.
dMulti0  <- rmultinom(1, size = 500, prob = p0)
dMulti1  <- rmultinom(1, size = 500, prob = p1)
sample0  <- rep(states, dMulti0)
sample1  <- rep(states, dMulti1)
sample0  <- factor(sample0, levels = states, ordered = T)
sample1  <- factor(sample1, levels = states, ordered = T)

# Munge simulated data.
data           <- rbind(data.frame("x" = 0, "y" = sample0),
                        data.frame("x" = 1, "y" = sample1))
data

## Run an unconstrained PO model
backend <- 'cmdstan'
f <- blrm(y~x, ppo=~x,data=data, backend = "cmdstan",keepsep='x', conc = 1,
          priorsd = 1, priorsdppo = 1, seed = 1234, iter = 2000,
          chains = 4,
          sampling.args = if(backend == 'rstan')
            list(control=list(adapt_delta=0.99,
            max_treedepth=12)))
