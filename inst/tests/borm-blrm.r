## Cross-checks borm() against blrm() on the logistic link, which is the
## one family blrm() implements.  borm() and blrm() are independently
## coded (separate Stan files -- ormcon.stan vs lrmconppo.stan -- and
## separate R data-prep/draws-extraction pipelines, though borm()'s
## pipeline closely mirrors blrm()'s), so close agreement between them
## is evidence for borm()'s correctness beyond comparing it to
## rms::orm() alone.  Self-contained: does not depend on anything else
## having been run first.
##
## Claude Sonnet 5 2026-11-15          entire file

require(rmsb)
cstanSet()

## ----------------------------------------------------------------------
## 1. No cluster, method='optimizing' (near-MLE for both)
##
## blrm()'s near-MLE setting is iprior=1 (flat/t prior on intercepts);
## borm() has no iprior choice (Dirichlet-only, per CLAUDE.md), so its
## own near-MLE setting instead temporarily forces conc=1 (a uniform
## Dirichlet) during optimizing.  These are two DIFFERENT approximations
## to the true MLE, so coefficients should agree closely but not
## exactly.  Also note: borm()'s reported deviance includes the uniform
## Dirichlet's own (nonzero) log-density normalizing constant, which
## iprior=1's flat prior does not have, so the two deviances are
## expected to differ by that constant -- not a sign of disagreement
## about the actual fit.
## ----------------------------------------------------------------------

set.seed(42)
n <- 200
x <- rnorm(n)
y <- as.integer(cut(0.8 * x + rlogis(n), breaks = c(-Inf, -1, 0, 1, Inf)))
d <- data.frame(x = x, y = y)
dd <- datadist(d); options(datadist = "dd")

f_blrm <- blrm(y ~ x, data = d, method = "optimizing", iprior = 1, showopt = FALSE)
f_borm <- borm(y ~ x, data = d, family = "logistic", method = "optimizing", showopt = FALSE)

cat("\n=== 1. No cluster, optimizing ===\n")
cat("blrm (iprior=1, near-MLE):\n")
print(f_blrm$coefficients)
cat("borm (conc forced to 1 internally, near-MLE):\n")
print(f_borm$coefficients)
cat("max|coefficient diff|:",
    max(abs(f_blrm$coefficients - f_borm$coefficients[names(f_blrm$coefficients)])),
    "  (expect small, e.g. < 1e-3 -- not exactly 0, see comment above)\n")
cat("blrm deviance:", f_blrm$deviance, " borm deviance:", f_borm$deviance,
    " (expected to differ by the Dirichlet(1,...,1) log-normalizing constant)\n")

## ----------------------------------------------------------------------
## 2. cluster(), single-sigma random intercept, real NUTS sampling
##
## The more informative check: both fits use actual MCMC (not an
## optimizer) on the identical data and the same sampling.args seed, so
## close agreement in both the fixed-effect coefficients AND sigmag
## (the random-intercept SD) is evidence the two independently-coded
## hierarchical-model Stan files and R pipelines are estimating the same
## thing.
## ----------------------------------------------------------------------

set.seed(31)
nid <- 30; nper <- 4; n2 <- nid * nper
id <- rep(1 : nid, each = nper)
x2 <- rnorm(n2)
re <- rnorm(nid, 0, 0.8)[id]
y2 <- as.integer(cut(0.6 * x2 + re + rlogis(n2), breaks = c(-Inf, -1, 0, 1, Inf)))
d2 <- data.frame(x = x2, y = y2, id = id)

f_blrm2 <- blrm(y ~ x + cluster(id), data = d2, iter = 1000, warmup = 500,
                chains = 2, refresh = 0, sampling.args = list(seed = 99))
f_borm2 <- borm(y ~ x + cluster(id), data = d2, family = "logistic",
                iter = 1000, warmup = 500, chains = 2, refresh = 0,
                sampling.args = list(seed = 99))

cat("\n=== 2. cluster(), single-sigma, real NUTS sampling ===\n")
cat("blrm posterior mean coefficients:\n")
print(colMeans(f_blrm2$draws))
cat("borm posterior mean coefficients:\n")
print(colMeans(f_borm2$draws))
cat("blrm sigmag posterior mean:", mean(f_blrm2$omega[, "sigmag"]), "\n")
cat("borm sigmag posterior mean:", mean(f_borm2$omega[, "sigmag"]), "\n")
cat("(expect close agreement throughout, within ordinary MCMC noise for",
    "iter=1000, chains=2 on the same data -- not exact equality)\n")
