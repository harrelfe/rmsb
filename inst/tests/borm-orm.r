## Cross-checks borm() against rms::orm() (and, for the mix_re random-
## intercept extension, against rms::orm()'s own cluster()+mix_re()
## path, i.e. ormrfit()'s AGQ MLE) across all 5 link families borm()
## implements.  Self-contained: does not depend on anything else having
## been run first.
##
## Claude Sonnet 5 2026-11-15          entire file

require(rmsb)
cstanSet()

## ----------------------------------------------------------------------
## 1. All 5 link families, uncensored, through borm() itself
##
## method='optimizing' with conc temporarily forced to 1 (borm()'s own
## near-MLE setting, done internally) should closely match rms::orm()'s
## true MLE for every family, not just logistic.
## ----------------------------------------------------------------------

cat("\n=== 1. All 5 links, uncensored, borm(optimizing) vs orm MLE ===\n")
set.seed(11)
n <- 250
x <- rnorm(n)
for (fam in c("logistic", "probit", "loglog", "cloglog", "cauchit")) {
  u <- runif(n)
  eps <- switch(fam,
    logistic = qlogis(u), probit = qnorm(u),
    loglog = -log(-log(u)), cloglog = log(-log(1 - u)), cauchit = qcauchy(u)
  )
  y <- as.integer(cut(0.7 * x + eps,
    breaks = quantile(0.7 * x + eps, seq(0, 1, .2)), include.lowest = TRUE
  ))
  d <- data.frame(x = x, y = y)
  f_orm <- orm(y ~ x, data = d, family = fam)
  dd <- datadist(d); options(datadist = "dd")
  f_borm <- borm(y ~ x, data = d, family = fam, method = "optimizing", showopt = FALSE)
  cat(sprintf("%-10s max|coefficient diff| = %.4g  (expect small, e.g. < 1e-3)\n",
      fam, max(abs(coef(f_orm) - f_borm$coefficients[names(coef(f_orm))]))))
}

## ----------------------------------------------------------------------
## 2. Logistic, mixed left/right/interval censoring, through borm()
## ----------------------------------------------------------------------

cat("\n=== 2. Logistic, mixed censoring, borm(optimizing) vs orm MLE ===\n")
set.seed(21)
n2 <- 160
x2 <- rnorm(n2)
yy <- round(0.5 * x2 + rnorm(n2), 2)
a <- yy; b <- yy
rc <- sample(1:n2, 15); b[rc] <- Inf
rem <- setdiff(1:n2, rc); lc <- sample(rem, 15); a[lc] <- -Inf
rem2 <- setdiff(rem, lc); ic <- sample(rem2, 15); a[ic] <- yy[ic] - 0.3; b[ic] <- yy[ic] + 0.3
Y <- Ocens(a, b)
d2 <- data.frame(x = x2)
f_orm2 <- orm(Y ~ x2, data = d2)
f_borm2 <- borm(Y ~ x2, data = d2, method = "optimizing", showopt = FALSE)
cat("max|coefficient diff| =", max(abs(coef(f_orm2) - f_borm2$coefficients)),
    " (expect small, e.g. < 1e-3)\n")

## ----------------------------------------------------------------------
## 3. mix_re: borm()'s posterior mean for sigma1/sigma2 vs
##    rms::orm()'s cluster()+mix_re() path (ormrfit()'s AGQ MLE)
##
## This compares two independent estimation methods (AGQ marginal MLE
## vs. Bayesian MCMC) on the SAME data, not either one against the
## simulation's true sigma1/sigma2 -- with only 40 clusters estimating
## two variance components, a single realized dataset's point
## estimates are not expected to recover the true generating values
## closely, but the two METHODS should agree with each other if both
## are implementing the same model correctly. Needs enough iterations/
## chains for sigma1/sigma2 to actually converge (confirmed via Rhat
## below) -- an earlier, shorter attempt (iter=1500, chains=2) on a
## larger (80-cluster) version of this same simulation gave Rhat=1.83
## (useless, unconverged) results and was discarded; this configuration
## was confirmed to converge cleanly (Rhat ~1.00-1.002) before relying
## on it here.
## ----------------------------------------------------------------------

cat("\n=== 3. mix_re: borm() posterior mean vs rms::orm() AGQ (ormrfit) ===\n")
set.seed(17)
nid <- 40; nper <- 4; n3 <- nid * nper
id <- rep(1:nid, each = nper)
x3 <- rnorm(n3)
sigma1_true <- 0.7; sigma2_true <- 1.1
v <- rnorm(nid)
mre <- ave(rep(1:nper, nid), id, FUN = function(t) as.numeric(seq_along(t) > 1))
w <- sigma1_true * (1 - mre) + sigma2_true * mre
eta <- 0.5 * x3 + w * v[id]
y3 <- as.integer(cut(eta + rlogis(n3), breaks = c(-Inf, -1, 0, 1, Inf)))
d3 <- data.frame(x = x3, y = y3, id = id, mre = mre)

f_orm3 <- orm(y3 ~ x + cluster(id) + mix_re(mre), data = d3)
cat("orm (ormrfit AGQ MLE)   sigma1 =", f_orm3$sigma1, " sigma2 =", f_orm3$sigma2, "\n")
cat("true simulation values  sigma1 =", sigma1_true, " sigma2 =", sigma2_true,
    " (for reference only -- NOT the comparison this test is about)\n")

dd <- datadist(d3); options(datadist = "dd")
f_borm3 <- borm(y3 ~ x + cluster(id) + mix_re(mre), data = d3,
                iter = 3000, warmup = 1500, chains = 4, refresh = 0,
                sampling.args = list(seed = 7))
cat("borm (posterior mean)   sigma1 =", mean(f_borm3$omega[, "sigma1"]),
    " sigma2 =", mean(f_borm3$omega[, "sigma2"]), "\n")
cat("borm Rhat for sigma1/sigma2 (should be close to 1.00, confirming convergence):\n")
## diagnostics has a different shape per backend: a plain matrix for
## rstan, a list (with a $summary data frame) for cmdstan
if (f_borm3$backend == "cmdstan") {
  s <- f_borm3$diagnostics$summary
  print(s[s$variable %in% c("sigma1[1]", "sigma2[1]"), c("variable", "rhat", "ess_bulk", "ess_tail")])
} else {
  print(f_borm3$diagnostics[c("sigma1[1]", "sigma2[1]"), ])
}
cat("(expect orm's AGQ estimates and borm's posterior means to agree with EACH",
    "OTHER reasonably closely -- not necessarily with sigma1_true/sigma2_true above)\n")
