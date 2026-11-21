## Cross-checks rmsb::Survival.borm()/survest.borm() against
## rms::Survival.orm()/survest.orm() on a single simulated, right-censored
## dataset: compares both the point estimate (borm's posterior mean vs
## orm's MLE-based survival estimate) and the uncertainty interval (borm's
## highest posterior density interval vs orm's delta-method/Wald interval)
## at a handful of explicit times, for several covariate settings, plus
## the full default curve (every modeled intercept). Section 3 repeats
## the comparison for a model with a Y-dependent (non-proportional-odds)
## effect for x1, via yde/ytrans. Self-contained: does not depend on
## anything else having been run first.
##
## Claude Sonnet 5 2026-11-21          entire file

require(rmsb)
cstanSet()

cat("\n=== Simulating a single right-censored dataset ===\n")
set.seed(31)
n <- 400
x1 <- rnorm(n)
x2 <- rnorm(n)
true_time <- rexp(n, rate = 0.08 * exp(0.3 * x1 - 0.2 * x2))
cens_time <- runif(n, 4, 20)
obs_time  <- round(pmin(true_time, cens_time), 2)
is_cens   <- true_time > cens_time
cat("n =", n, " censored:", sum(is_cens), "/", n, "\n")

y <- Ocens(obs_time, ifelse(is_cens, Inf, obs_time))
d <- data.frame(x1 = x1, x2 = x2)

cat("\n=== Fitting rms::orm() (MLE) and rmsb::borm() (posterior) ===\n")
dd <- datadist(d); options(datadist = "dd")
f_orm <- orm(y ~ x1 + x2, data = d, y = TRUE)
f_orm

f_borm <- borm(y ~ x1 + x2, data = d, y = TRUE, method = "sampling",
               chains = 4, iter = 2000, refresh = 0,
               sampling.args = list(seed = 1))
f_borm

nd <- data.frame(x1 = c(-1, 0, 1), x2 = c(0.5, 0, -0.5))
times <- c(2, 5, 10, 15)

## ----------------------------------------------------------------------
## 1. Point estimates and intervals at explicit times
## ----------------------------------------------------------------------

cat("\n=== 1. Point estimates + intervals at explicit times ===\n")
r_orm  <- survest(f_orm,  newdata = nd, times = times, conf.int = 0.95)
r_borm <- survest(f_borm, newdata = nd, times = times, conf.int = 0.95)

cmp <- data.frame(
  Xrow        = r_orm$Xrow,
  time        = r_orm$time,
  surv_orm    = r_orm$surv,
  surv_borm   = r_borm$surv,
  surv_diff   = r_borm$surv - r_orm$surv,
  lower_orm   = r_orm$lower,
  lower_borm  = r_borm$lower,
  upper_orm   = r_orm$upper,
  upper_borm  = r_borm$upper,
  width_orm   = r_orm$upper  - r_orm$lower,
  width_borm  = r_borm$upper - r_borm$lower
)
print(round(cmp, 4))
cat("\nmax|surv_borm - surv_orm| =", max(abs(cmp$surv_diff), na.rm = TRUE),
    " (expect small -- posterior mean vs MLE on the same data)\n")
cat("borm HPD brackets its own point estimate:",
    all(r_borm$surv >= r_borm$lower - 1e-9 & r_borm$surv <= r_borm$upper + 1e-9,
        na.rm = TRUE), "\n")
cat("orm Wald CI brackets its own point estimate:",
    all(r_orm$surv >= r_orm$lower - 1e-9 & r_orm$surv <= r_orm$upper + 1e-9,
        na.rm = TRUE), "\n")
cat("interval widths broadly comparable (ratio borm/orm, median):",
    median(cmp$width_borm / cmp$width_orm, na.rm = TRUE),
    " (expect roughly near 1 -- HPD under a weak prior vs Wald on the",
    "same n, not an exact match)\n")

## ----------------------------------------------------------------------
## 2. Full default curve (every modeled intercept), one covariate setting
## ----------------------------------------------------------------------

cat("\n=== 2. Full default curve (times unspecified), one covariate setting ===\n")
nd1 <- nd[1, , drop = FALSE]
rc_orm  <- survest(f_orm,  newdata = nd1, conf.int = 0.95)
rc_borm <- survest(f_borm, newdata = nd1, conf.int = 0.95)
cat("nrow orm:", nrow(rc_orm), " nrow borm:", nrow(rc_borm), "\n")

## orm and borm place their intercepts at the same distinct Y values, so
## the two curves should line up time-for-time; confirm before comparing.
stopifnot(isTRUE(all.equal(rc_orm$time, rc_borm$time)))
dmax <- max(abs(rc_borm$surv - rc_orm$surv), na.rm = TRUE)
cat("max|surv_borm - surv_orm| over the full curve =", dmax,
    " (expect small)\n")
cat("both curves monotonic non-increasing:",
    all(diff(rc_orm$surv[!is.na(rc_orm$surv)])  <= 1e-9),
    all(diff(rc_borm$surv[!is.na(rc_borm$surv)]) <= 1e-9), "\n")

## ----------------------------------------------------------------------
## 3. Y-dependent (non-proportional-odds) effect for x1, via yde/ytrans
##
## Y is coarsened for this section only (sections 1/2 above are
## unaffected, still fitted on the original, unbinned y) -- an x1*yt
## interaction this strong, combined with ~400 observations spread over
## 208 distinct Y values, was found by direct investigation to produce
## a genuinely ill-conditioned posterior: the Hessian at the optimum has
## an eigenvalue of -2.08e-05 (vs. -0.26 for the same data WITHOUT yde),
## with the corresponding eigenvector loaded 99.9% onto a single
## category probability for a Y level with exactly one exact event.
## Confirmed this is a general sparse-cell problem, not a right-tail
## phenomenon specifically: right-censoring everything beyond the 3rd
## highest uncensored Y barely changed k (205 vs 208, since that Y level
## itself was in the middle of the range, not the tail) and made
## divergences WORSE (74% vs 65%). What did work: coarsening so every
## resulting Y level has a guaranteed minimum number of EXACT events
## (not Hmisc::cut2()'s own default of total observations including
## censored ones, which was confirmed to still leave zero-exact-event
## bins). Swept the minimum (m) across 3 seeds each at the full
## chains=4/iter=2000 settings actually used below, to find the most
## resolution obtainable without going much above 1% divergent
## transitions (not just a single lucky run): m=6-7 (k=38,33) held at a
## consistent 2%; m=8 (k=28) was borderline (1-2% across seeds); m=9
## (k=25) was the smallest m giving a seed-consistent 1% across all 3
## seeds tried, so it is used here. (A few percent divergence, not
## clustered in one region, is within the range Stan's own guidance
## treats as not necessarily indicating a real problem -- this is a
## large, qualitative improvement from the original k=208's 65%, not a
## claim of exactly zero.) Pushing m much higher (15-25) does not
## improve on this and was sometimes worse, so m=9 is not a value
## chosen to look clean; it's the result of a real investigated
## resolution/stability tradeoff documented in the session that
## produced this test. NO adapt_delta/max_treedepth override is used.
##
## ytrans centers Y at the mean of the UNCENSORED, COARSENED observations
## (the conditioning rms::orm()/rmsb::borm() themselves document for
## ytrans -- see orm-yde-implementation.md Section 12), not the overall
## (censored + uncensored) y, since Inf values would otherwise
## contaminate the mean.
##
## rms::ormYdeSurvival() does not yet support conf.int (an explicit,
## documented stop()), so only point estimates are compared against
## rmsb's yde-aware Survival.borm(); borm's own HPD interval is checked
## only for self-consistency (brackets its own point estimate).
##
## The orm side calls Survival(f_orm_yde)(newdata=..., times=...)
## directly rather than survest(f_orm_yde, ...): confirmed directly that
## rms::survest.orm() has a pre-existing bug for a yde fit -- it derives
## linear.predictors=NULL (never reassigning it once newdata is given)
## and then calls S(times, linear.predictors, X=x, ...), which passes
## that NULL into the SECOND positional argument of whatever Survival()
## returns. For a non-yde fit that argument is lp (harmless, since X
## overrides it), but for a yde fit Survival.orm() returns
## ormYdeSurvival()'s own closure, whose second positional argument is
## newdata -- so newdata itself gets clobbered to NULL, and the call
## fails with ormYdeSurvival()'s own "must be called with newdata"
## error. rmsb's survest.borm() does not share this bug (it passes
## newdata=newdata by name, not positionally), so the borm side below
## still goes through survest(). This is an rms-side finding to report
## separately, not fixed here per CLAUDE.md (rms is read-only; a fix
## needs Frank's go-ahead).
## ----------------------------------------------------------------------

cat("\n=== 3. yde/ytrans: Y-dependent effect for x1 ===\n")

## Build Y levels with a guaranteed minimum number of EXACT events per
## level (not Hmisc::cut2()'s own default of total observations, which
## can still leave zero-exact-event levels -- confirmed directly). Break
## points come from the uncensored times alone (via cut2's own
## onlycuts=TRUE, which returns the numeric cutpoints directly rather
## than interval-label factor levels that would need text parsing);
## censored times are then assigned to whichever resulting interval
## they fall into, and anything beyond the last break (deep
## right-censored tail) maps to the last interval's own representative
## value.
bormTargetedY <- function(obs_time, is_cens, m) {
  ut   <- obs_time[! is_cens]
  brks <- Hmisc::cut2(ut, m = m, onlycuts = TRUE)
  brks[1] <- -Inf; brks[length(brks)] <- Inf
  cuts    <- cut(obs_time, breaks = brks, include.lowest = TRUE, right = FALSE)
  cuts_ut <- cut(ut,       breaks = brks, include.lowest = TRUE, right = FALSE)
  repval  <- tapply(ut, cuts_ut, mean)
  newy    <- repval[as.character(cuts)]
  newy[is.na(newy)] <- max(repval, na.rm = TRUE)
  as.numeric(newy)
}

obs_time_yde <- bormTargetedY(obs_time, is_cens, m = 9)
k_yde <- length(unique(obs_time_yde))
cat("coarsened Y for this section: k =", k_yde,
    " (vs", length(f_orm$yunique), "distinct Y for sections 1/2's unbinned y)\n")

y_yde <- Ocens(obs_time_yde, ifelse(is_cens, Inf, obs_time_yde))

ybar <- mean(obs_time_yde[! is_cens])
yt <- function(y) y - ybar
formals(yt)$ybar <- ybar
cat("ybar (mean of coarsened uncensored y) =", ybar, "\n")

f_orm_yde <- orm(y_yde ~ x1 + x2, yde = ~ x1, ytrans = yt, data = d, y = TRUE)
f_orm_yde

f_borm_yde <- borm(y_yde ~ x1 + x2, yde = ~ x1, ytrans = yt, data = d, y = TRUE,
                   method = "sampling", chains = 4, iter = 2000, refresh = 0,
                   sampling.args = list(seed = 1))
f_borm_yde

cat("\n--- 3a. Point estimates at explicit times ---\n")
ry_orm  <- Survival(f_orm_yde)(times = times, newdata = nd, forcedf = TRUE)
ry_borm <- survest(f_borm_yde, newdata = nd, times = times, conf.int = 0.95)

cmpy <- data.frame(
  Xrow       = ry_borm$Xrow,
  time       = ry_borm$time,
  surv_orm   = ry_orm$surv,
  surv_borm  = ry_borm$surv,
  surv_diff  = ry_borm$surv - ry_orm$surv,
  lower_borm = ry_borm$lower,
  upper_borm = ry_borm$upper
)
print(round(cmpy, 4))
cat("\nmax|surv_borm - surv_orm| (yde) =", max(abs(cmpy$surv_diff), na.rm = TRUE),
    " (expect small -- posterior mean vs MLE on the same data)\n")
cat("borm HPD brackets its own point estimate (yde):",
    all(ry_borm$surv >= ry_borm$lower - 1e-9 & ry_borm$surv <= ry_borm$upper + 1e-9,
        na.rm = TRUE), "\n")

cat("\n--- 3b. Full default curve (times unspecified), one covariate setting ---\n")
## zero=TRUE (prepending a time=0/surv=1 row) is what survest.borm() always
## passes internally; matched here so the two curves line up row-for-row.
rcy_orm  <- Survival(f_orm_yde)(newdata = nd1, forcedf = TRUE, zero = TRUE)
rcy_borm <- survest(f_borm_yde, newdata = nd1, conf.int = 0.95, forcedf = TRUE)
stopifnot(isTRUE(all.equal(rcy_orm$time, rcy_borm$time)))
dmaxy <- max(abs(rcy_borm$surv - rcy_orm$surv), na.rm = TRUE)
cat("max|surv_borm - surv_orm| (yde, full curve) =", dmaxy, " (expect small)\n")
cat("both curves monotonic non-increasing (yde):",
    all(diff(rcy_orm$surv[!is.na(rcy_orm$surv)])   <= 1e-9),
    all(diff(rcy_borm$surv[!is.na(rcy_borm$surv)])  <= 1e-9), "\n")

cat("\nDONE\n")
