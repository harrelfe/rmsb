##' Bayesian Ordinal Regression
##'
##' Uses `rstan` with pre-compiled Stan code, or `cmdstan`, to get posterior draws of parameters from a Bayesian ordinal (or binary) cumulative probability regression model, with a choice of 5 link families -- `borm()` is to [blrm()] as [rms::orm()] is to [rms::lrm()]: it generalizes the single (logistic) link that `blrm()` implements to all of the link families `orm()` supports, and is designed, like `orm()`, to scale to a very large number of distinct Y values.  Unlike `blrm()`, `borm()` does not implement the partial proportional odds model (`ppo`/`cppo`) itself; `blrm()` remains the function to use for that. Instead `borm()` implements a more general Y-dependent effect mechanism that plays `cppo`'s role, via two new arguments: `yde` (paralleling `ppo`) identifies which predictors get a Y-dependent effect, and `ytrans` (paralleling `cppo`, but returning a matrix of one or more columns instead of a single score) specifies the shape(s) of that dependency. A single-column `ytrans` with a single `yde` predictor reduces mathematically to `blrm()`'s own `cppo` mechanism.
##'
##' Y-dependent effects of selected predictors can also be included in the model by the user providing the `yde` and `ytrans` arguments. For a logit link this implements the constrained partial proportional odds model of Peterson and Harrell (1990). For a log-log link this represents time-dependent covariate effects in a Cox proportional hazards model when Y=time.
##'
##' The Stan code internally uses a Dirichlet prior on the multinomial (category) probabilities at the predictor means, inducing a prior for the intercepts, exactly as [blrm()]'s own default (`iprior=0`) does -- `borm()` does not offer `blrm()`'s other two intercept-prior choices, since simulations showed they slow sampling badly once the number of distinct Y values is large, the regime `borm()` is specifically meant for.  Likewise, no data-dependent conditionals appear in the Stan code at all: censoring is handled entirely through subscripts precomputed once in R by [rms::ormIdx()] (the same routine [rms::orm.fit()] uses), not through per-observation branching in Stan.
##'
##' Design matrix columns are centered and QR-orthogonalized (via `rmsb`'s internal `selectedQr()`, the identical mechanism [blrm()] uses) before being handed to the Stan code, so that highly collinear columns do not hinder posterior sampling; **this transformation is done entirely in R, not in the Stan code**, which receives and fits `beta` on the already-transformed scale and knows nothing of the original covariate scale. Parameters are transformed back to the original data scale before being returned. Stan diagnostic output (e.g. from [stanDx()] or [stanGet()]) will show intercepts and coefficients on the centered/QR-transformed scale; everything `borm()` itself returns and prints is on the original, uncentered scale.
##'
##' As with [blrm()], priors on regression coefficients (`beta`) are specified through contrasts, via `pcontrast`, rather than directly; see [blrm()]'s own documentation for the full `pcontrast` syntax, which `borm()` reuses unchanged (contrasts on `beta` alone). `tau` (the `yde` coefficient matrix, present only when `yde`/`ytrans` are given) gets a single default prior, `normal(0, tausd)` independently on every entry, which `npcontrast` can override/augment for specific contrasts -- but unlike `blrm()`'s own `npcontrast` (which can only reference `tau`), `borm()`'s `npcontrast` can reference `beta`, `tau`, or both in the same contrast, since a Y-dependent effect is a combination of the two (`beta_j + Z_j . tau . ytransmat[y,]`, the effect of predictor `j` at Y value `y`). Each named predictor-setting list in `npcontrast` may include `ycut`, the Y value at which to evaluate that setting's `tau` contribution (see `pcontrast`'s own documentation for the general named-setting-list/`contrast=expression()` syntax, which is otherwise identical); a setting with no `ycut` contributes a pure-`beta` row, letting simple contrasts reduce to exactly the pure-`beta` or pure-`tau` case with no special syntax -- e.g. two settings sharing the same covariate values but different `ycut` have their `beta` parts cancel exactly in a difference, giving a contrast purely on the Y-dependent effect.
##'
##' `borm()` implements a single-level hierarchical random intercept for repeated-measures/clustered data via `cluster()` in the model formula, exactly as [blrm()] does, and additionally implements the dual-sigma `mix_re` extension to that random intercept (from `orm.rfit()`, `rms`'s internal, undocumented random-intercept fitter) via a `mix_re()` term in the formula (used together with `cluster()`), exactly as [rms::orm()] does -- see [rms::orm.fit()]'s own documentation, under `mre`, for the full statistical description of this extension. `mix_re` is not a `borm()` argument; it is extracted from the formula the same way `cluster` is.
##'
##' When using the `cmdstan` backend, `cmdstanr` will need to compile the Stan code once per computer, only recompiling when the Stan source code changes. By default the compiled code is stored in directory `.rmsb` under your home directory. Specify `options(rmsbdir=)` to specify a different location.
##'
##' If you want to run MCMC sampling even when no inputs or Stan code have changed, i.e., to use a different random number seed for the sampling process when you did not specify `sampling.args(seed=...)`, remove the `file` before running `borm`.
##'
##' Set `options(rmsbmsg=FALSE)` to suppress certain information messages.
##' @param formula an R formula object that can use `rms` package enhancements such as the restricted interaction operator, and may include `cluster()` and (together with `cluster()`) `mix_re()` terms
##' @param yde formula specifying the model predictors for which a Y-dependent effect is to be estimated, with no left-hand-side variable, paralleling [blrm()]'s own `ppo` argument exactly (including that its predictors may, and usually will, overlap those in `formula`). Must be accompanied by `ytrans`. If a predictor in `yde` is not in `formula`, it contributes only a Y-dependent effect, no constant (proportional-odds) one.
##' @param ytrans a function of one argument, `y`, specifying the shape(s) of the Y-dependent effect(s) for the predictors in `yde`, paralleling [blrm()]'s own `cppo` argument but generalized to allow more than one shape at once, or the character string `'all'` for a fully saturated (no-smoothness-assumption) Y-dependent effect: one free `tau` parameter per modeled intercept except the first (which is left at an implicit zero deviation, needed for identifiability when a `yde` predictor is also in `formula`, the usual case), i.e. `k-1` free parameters per `yde` predictor for `k` modeled intercepts -- matching [rms::orm()]'s own identical `ytrans='all'` shortcut exactly (same construction, same anchoring, same `k <= 50` and no-censoring restrictions below). `ytrans='all'` does not support censored Y, and requires `k <= 50` (the number of distinct Y values, for computational reasons). Otherwise, when `ytrans` is a function: it is called on the distinct, ordered Y values seen in the data (`y`'s own values need not be the complete set, or in any particular order -- `ytrans` must work correctly on an arbitrary subset/order of Y values, since it is also called later, e.g. for contrasts, on individual Y values) and must return either a plain vector (one Y-dependent shape) or a matrix with one or more columns (several simultaneous Y-dependent shapes), one row (or value) per element of `y`. `ytrans` is used exactly as written for every purpose a user or `borm()`'s own methods will see (storage, `bormStats()`, `predict.borm()`), with no centering -- so `tau`, the coefficient `borm()` reports, is directly interpretable as "per unit of `ytrans(y)`, per unit of the `yde` predictor," independent of the sample means of `y`, the `yde` predictors, or any other covariate; an earlier version centered `ytrans` automatically, which was reversed after confirming directly that it silently distorted the *other*, ordinary (non-`yde`) coefficient(s) for any predictor appearing in both `yde` and the main formula -- the common case, since a Y-dependent effect is normally an extension of an ordinary effect, not a separate one -- while providing no measurable sampling benefit. Internally (invisible to the user, and not reflected in the stored/reported `tau`), `ytrans(y)` IS still divided by its own SD over the observed Y values before being given to Stan, purely for numerical safety -- confirmed directly that skipping this is not safe: with no scaling at all, a `ytrans` producing values in roughly the hundreds or more (`ytrans=function(y) 100*y` on data with `y` up to about 10, for example) crashed R outright (a Stan optimizer segfault), while values up to the tens did not. A single-column `ytrans` applied to a single `yde` predictor reduces mathematically to `blrm()`'s own `cppo` mechanism. Must be accompanied by `yde`. When `ytrans` is a user-specified function, it is recommended to mean-center any column it produces, using that column's own sample mean, when the uncentered column's values have a mean far from zero. See the Examples section below, where the means are computed and stored for later use. When `ytrans` produces column names, these are used as part of the names given to the `tau` parameters.
##' @param data a data frame; defaults to using objects from the calling environment
##' @param subset a logical vector or integer subscript vector specifying which subset of data should be used
##' @param na.action default is `na.delete` to remove missings and report on them
##' @param family a character value specifying the distribution family, corresponding to logistic (the default), probit, loglog (`exp(-exp(-x))`; extreme value type I/Gumbel maximum), cloglog (`1-exp(-exp(x))`; Gumbel minimum), and cauchit (Cauchy) link functions for `Prob[Y >= y | X]`, matching [rms::orm()]'s own `family` argument exactly
##' @param tausd the prior SD for a `normal(0, tausd)` prior placed independently on every entry of `tau`, the `yde` coefficient matrix (one row per `yde` predictor, one column per `ytrans` column); ignored when `yde`/`ytrans` are not given. Specified on a RAW, interpretable, dataset-invariant basis -- one unit of a `yde` predictor, one SD of `ytrans(y)` over the observed Y values -- not on the internal (QR-rotated, further-scaled) basis `tau` is actually sampled on; `borm()` converts it internally. This matters because the OLD convention (applying `tausd` directly as the internal SD) was not portable across datasets: the same number implied a very different effective prior depending on `y`'s own raw units, confirmed directly by comparing datasets with `y` on scales differing by 100-fold. The default, 10, is a round, moderate choice on this new basis, clear of the (robustly confirmed) region below about 1 where `tau` gets visibly shrunk toward 0 -- it is NOT the result of tuning against NUTS sampling efficiency (leapfrog/treedepth counts): those were found, on closer inspection, to be chaotically sensitive to `tausd` at a resolution far finer than any such tuning could resolve (confirmed directly: a 0.001% change in `tausd` swung total leapfrog steps between roughly 200,000 and 350,000, reproducibly), so no specific number can honestly be presented as minimizing them. For `qyde=1` (every case validated here) the raw-to-internal conversion is exact; for more than one `yde` predictor it uses a reasonable but not independently validated generalization (see the code comment at its computation).
##' @param conc the Dirichlet distribution concentration parameter for the prior distribution of cell probabilities at covariate means, with the same default as [blrm()]'s own `conc` argument: the reciprocal of 0.8 + 0.35 max(k, 3) where k is the number of Y categories. For optimizing, the concentration parameter is always 1.0, to obtain results close to the MLE for providing the posterior mode.
##' @param psigma defaults to 1 for a half-t distribution with 4 d.f., location parameter `rsdmean` and scale parameter `rsdsd`, for the random-intercept SD (or, when `mix_re` is used, for `sigma1`). Set `psigma=2` to use the exponential distribution instead.
##' @param rsdmean the assumed mean of the prior distribution of the random-intercept SD (or of `sigma1` when `mix_re` is used). When `psigma=2` this is the mean of an exponential distribution and defaults to 1. When `psigma=1` this is the mean of the half-t distribution and defaults to zero.
##' @param rsdsd applies only to `psigma=1` and is the scale parameter for the half-t distribution for the random-intercept SD (or `sigma1`), defaulting to 1. When `mix_re` is used, `rsdsd` is also reused, regardless of `psigma`, as the scale of a `normal(0, rsdsd)` prior for `sigma2` (which, unlike `sigma1`, may be negative) -- a reasonable default rather than something `orm.rfit()` (rms internal) itself specifies, since that function is a pure MLE with no prior on `sigma2` at all.
##' @param pcontrast a list specifying contrasts that are to be given Gaussian prior distributions, with exactly the syntax and semantics of [blrm()]'s own `pcontrast` argument; see its documentation for full details and examples.
##' @param npcontrast a list specifying contrasts on `tau` (the `yde` coefficient matrix), `beta`, or both, to be given Gaussian prior distributions, with the same named-predictor-setting-list/`contrast=expression()` syntax as `pcontrast` (see [blrm()]'s own `pcontrast` documentation), plus an optional `ycut` -- a single Y value applied to every named setting, or a named vector/list giving a different Y value per named setting -- specifying the Y value at which each setting's `tau` contribution is evaluated; a setting with no `ycut` contributes a pure-`beta` row. Requires `yde`/`ytrans` to be given; it is an error to supply `npcontrast` otherwise.
##' @param backend set to `cmdstan` to use `cmdstan` through the R `cmdstanr` package instead of the default `rstan`. You can also specify this with a global option `rmsb.backend`.
##' @param iter number of posterior samples per chain for [rstan::sampling()] to run, counting warmups
##' @param warmup number of warmup iterations to discard. Default is `iter`/2.
##' @param chains number of separate chains to run
##' @param refresh see [rstan::sampling()] and [cmdstanr::sample()]. The default is 0, indicating that no progress notes are output. If `refresh > 0` and `progress` is not `''`, progress output will be appended to file `progress`.
##' @param progress see `refresh`. Defaults to `''` if `refresh = 0`.
##' @param x set to `FALSE` to not store the design matrix in the fit. `x=TRUE` is needed if running `bormStats`/`print.borm` for example -- when `yde`/`ytrans` are also given, `x=TRUE` additionally stores `z`, the `yde` design matrix, which `bormStats` also needs.
##' @param y set to `FALSE` to not store the response variable in the fit
##' @param loo set to `FALSE` to not run `loo` and store its result as object `loo` in the returned object. `loo` defaults to `FALSE` if the sample size is greater than 1000.
##' @param ppairs set to a file name to run `rstan` `pairs` or, if `backend='cmdstan'` `bayesplot::mcmc_pairs` and store the resulting png plot there. Set to `TRUE` instead to directly plot these diagnostics.
##' @param method set to `'optimizing'` to run the Stan optimizer and not do posterior sampling, `'both'` (the default) to run both the optimizer and posterior sampling, or `'sampling'` to run only the posterior sampling. Running `optimizing` is a way to obtain maximum likelihood estimates. When `cluster()` is present in the formula, `method` is automatically set to `'sampling'` as maximum likelihood estimates without marginalizing over the random effects do not make sense.
##' @param inito initial value for optimization. The default is the `rstan` default `'random'`. Frequently specifying `inito=0` will benefit when the number of distinct Y categories grows.
##' @param inits initial value for sampling, defaults to `inito`
##' @param standata set to `TRUE` to return the Stan data list and not run the model
##' @param debug set to `TRUE` to output timing and progress information to /tmp/debug.txt
##' @param file a file name for a `saveRDS`-created file containing or to contain the saved fit object. If `file` is specified and the file does not exist, it will be created right before the fit object is returned, less the large `rstan` object. If the file already exists and its inputs are unchanged, the previously stored fit object is immediately returned and no new calculations are done.
##' @param sampling.args a list containing parameters to pass to [rstan::sampling()] or to the `cmdstanr` `sample` function, other than `iter, warmup, chains, refresh, init`, which are already arguments to `borm`. A good use of this is `sampling.args=list(seed=3)` to get reproducible sampling. If Stan reports many divergent transitions -- which can happen for near-separated data, e.g. two groups whose Y distributions barely overlap -- try `sampling.args=list(control=list(adapt_delta=0.995, max_treedepth=15))` (for `rstan`; `cmdstanr`'s `sample()` takes `adapt_delta`/`max_treedepth` directly rather than nested in `control`) before concluding the model itself is wrong: in one such case this cut divergent transitions by about 85% with no change to the model or data. This is not `borm()`'s own default because it roughly doubles sampling time on an otherwise well-behaved fit for no benefit (confirmed directly: 2.4x slower, identical zero divergences, on ordinary non-separated data) -- raise it only when you actually see divergence warnings.
##' @param showopt set to `TRUE` to show Stan optimizer output
##' @param cores_loo number of cores for [loo::loo()]'s own internal Pareto-smoothing computation (passed as its `cores` argument, for both backends). Defaults to `1`, not to the number of cores `options('mc.cores')` may be set to for other purposes (e.g. parallelizing chains), because `loo()` tries to fork the R session whenever `cores > 1`, and forking is unavailable in some environments -- e.g. the Positron IDE -- causing `loo()` to fail with an uninformative "loo failed" warning until that error is actually surfaced. Increase this only when you know forking is safe in your environment.
##' @param ... passed to [rstan::optimizing()] or the `cmdstanr` optimizing function. `sampling.args` is usually used instead.
##' @return an `rms` fit object of class `borm`, `rmsb`, `rms`, closely paralleling [blrm()]'s own return value (see its documentation); also contains `rstan` or `cmdstanr` results under the name `rstan`.
##' @examples
##' \dontrun{
##'   getHdata(titanic3)
##'   dd <- datadist(titanic3); options(datadist='dd')
##'   f <- borm(age ~ sex + pclass, data=titanic3, family='probit')
##'   f                   # model summary using print.borm
##'   coef(f)             # compute posterior mean parameter values
##'   stanDx(f)           # print basic Stan diagnostics
##'
##'   # Fit a random effects model to handle multiple observations per
##'   # subject ID, with rms::orm.rfit()'s mix_re extension
##'   # mre <- ave(time, id, FUN = function(t) as.numeric(t > min(t)))
##'   # f <- borm(y ~ x + cluster(id) + mix_re(mre), data=mydata)
##' }
##' \dontrun{
##'   # For Y=0, 1, 2 for no pain, pain, or death fit a partial proportional
##'   # odds model that allows for a special effect of x1 on death (the
##'   # linear part only)
##'   f <- borm(y ~ rcs(x1, 4) + x2, ~ x1, ytrans=function(y) 1 * (y == 2))
##'
##'   # Fit the equivalent of an unequal variance 2-sample t-test for
##'   # continuous Y.  This corresponds to a linear rescaling of probit
##'   # ordinal model intercepts
##'   yt <- function(y, bar) y - bar
##'   formals(yt)$bar <- mean(y)
##'   f <- borm(y ~ treatment, ~ treatment, ytrans=yt, family='probit')
##'
##'   # For continuous Y, let the linear effect of x1 be U-shaped over Y levels
##'   yt <- function(y, bar) cbind(lin=y - bar, quad=(y - bar) ^ 2)
##'   formals(yt)$bar <- mean(y)
##'   yt
##'   # Note that in the function definition, bar is now a fixed number,
##'   # not the expression mean(y)
##'   f <- borm(y ~ rcs(x1, 4) + x2, ~ x1, ytrans=yt)
##'
##'   # In some cases a vector of means will be needed:
##'   yt <- function(y, bar) cbind(a=fun1(y) - bar[1], b=fun2(y) - bar[2])
##'   formals(yt)$bar <- c(mean(fun1(y)), mean(fun2(y)))
##'
##'   # When using a spline function it may be advisable to use an
##'   # orthogonal basis that is nicely scaled, perhaps natural splines.
##' }
##' @author Frank Harrell
##' @seealso [blrm()], [rms::orm()], [rms::ormIdx()], [print.borm()], [bormStats()], [stanDx()], [stanGet()], [coef.rmsb()], [vcov.rmsb()], [print.rmsb()]
##' @export
##' @md
borm <- function(formula, yde = NULL, ytrans = NULL,
                 data = environment(formula), subset, na.action = na.delete,
                 family = c("logistic", "probit", "loglog", "cloglog", "cauchit"),
                 conc = 1. / (0.8 + 0.35 * max(k, 3)),
                 psigma = 1, rsdmean = if (psigma == 1) 0 else 1,
                 rsdsd = 1, tausd = 10, pcontrast = NULL, npcontrast = NULL,
                 backend = c("rstan", "cmdstan"),
                 iter = 2000, warmup = iter / 2, chains = 4, refresh = 0,
                 progress = if (refresh > 0) "stan-progress.txt" else "",
                 x = TRUE, y = TRUE, loo = n <= 1000, ppairs = NULL,
                 method = c("both", "sampling", "optimizing"),
                 inito = "random", inits = inito,
                 standata = FALSE, file = NULL, debug = FALSE,
                 sampling.args = NULL, showopt = FALSE, cores_loo = 1, ...) {
  call <- match.call()
  method <- match.arg(method)
  family <- match.arg(family)
  if (length(yde) && !length(ytrans)) stop("ytrans must be given when yde is given")
  if (length(ytrans) && !length(yde)) stop("yde must be given when ytrans is given")
  # Claude Sonnet 5 2026-11-22
  # ytrans='all' (a character string, not a function) is a special
  # request, substituted for a real function once k/anycens are known
  # -- see the identical(ytrans, 'all') block below. Matches the check
  # rms::orm.fit() already has for its own identical shortcut.
  if (length(ytrans) && !is.function(ytrans) && !identical(ytrans, "all"))
    stop("ytrans must be a function or 'all'")
  if (length(npcontrast) && !length(yde)) {
    stop("npcontrast requires yde/ytrans to be given (it specifies contrasts on tau)")
  }
  backend <- if (missing(backend)) {
    getOption("rmsb.backend", "rstan")
  } else {
    match.arg(backend)
  }

  if (backend == "cmdstan") {
    rmsbdir <- getOption("rmsbdir")
    if (!length(rmsbdir)) {
      rmsbdir <- "~/.rmsb"
      if (getOption("rmsbmsg", TRUE)) {
        message("Compiled Stan code will be stored in ~/.rmsb.  Use options(rmsbdir=) to override.")
      }
    }
    dir.create(rmsbdir, showWarnings = FALSE)
  }

  callenv <- parent.frame() # don't delay these evaluations
  subset <- if (!missing(subset)) eval(substitute(subset), data, callenv)

  data <- rms::modelData(data, formula, yde,
    subset = subset,
    na.action = na.action, callenv = callenv
  )
  # Claude Sonnet 5 2026-11-21          2 lines
  # data2 is rms::modelData()'s own second-formula-safe copy of data
  # (the response variable is not present in it), needed when calling
  # rms::Design() a second time for yde -- calling Design() twice on
  # the same data object fails ("program logic error tl"), confirmed
  # directly; blrm.r's own ppo block uses the identical data2 mechanism.
  data2 <- attr(data, "data2")
  attr(data, "data2") <- NULL

  prevhash <- NULL
  if (length(file) && file.exists(file)) {
    prevfit <- readRDS(file)
    prevhash <- prevfit$datahash
  }

  if (debug) {
    debug <- function(...) {
      cat(..., format(Sys.time()), "\n", file = "/tmp/debug.txt", append = TRUE)
    }
  } else {
    debug <- function(...) {}
  }

  nact <- NULL
  yname <- as.character(formula[2])
  en <- environment(formula)
  requireNamespace("rstan", quietly = TRUE)

  # Claude Sonnet 5 2026-11-19          1 line
  # mix_re() is a legitimate rms::Design() formula special (see
  # R/cluster.r and R/rms.s in rms, and rms::orm()'s own identical
  # specials=c("cluster","mix_re") call) -- not a borm() argument,
  # matching orm()'s own user-facing convention exactly.
  X <- rms::Design(data, formula = formula, specials = c("cluster", "mix_re"))

  cluster     <- attr(X, "cluster")
  clustername <- attr(X, "clustername")
  mix_re      <- attr(X, "mix_re")
  mix_rename  <- attr(X, "mix_rename")
  if (length(mix_re) && (length(mix_re) != length(cluster))) {
    stop("mix_re and cluster must have the same length")
  }
  use_mre <- length(mix_re) > 0
  if (use_mre && diff(range(mix_re)) == 0.0) {
    stop("mix_re has no variation (all values equal ", mix_re[1],
         "); sigma1 and sigma2 would not be separately identifiable")
  }

  atrx <- attributes(X)
  sformula <- atrx$sformula
  nact <- atrx$na.action
  Terms <- atrx$terms
  attr(Terms, "formula") <- formula
  atr <- atrx$Design
  mmcolnames <- atr$mmcolnames

  Y <- model.extract(X, "response")
  ## model.extract (which calls model.response) not keeping class
  isOcens <- inherits(Y, "Ocens") || NCOL(Y) == 2
  zero3 <- c(left = 0L, right = 0L, interval = 0L)
  Ncens1 <- Ncens2 <- zero3

  if (!isOcens) {
    Y <- Ocens2ord(Ocens(Y))
    ay <- attributes(Y)
    k <- length(ay$levels)
  } else {
    Y <- Ocens2ord(Y)
    ay <- attributes(Y)
    k <- length(ay$levels)
    if (length(ay$Ncens1)) Ncens1 <- ay$Ncens1
    if (length(ay$Ncens2)) Ncens2 <- ay$Ncens2
  }

  X <- model.matrix(Terms, X)
  alt <- attr(mmcolnames, "alt")
  if (!all(mmcolnames %in% colnames(X)) && length(alt)) mmcolnames <- alt
  X <- X[, mmcolnames, drop = FALSE]
  colnames(X) <- atr$colnames

  ## Get rid of row names and other attributes that rstan barks about
  trimit <- function(z) {
    attr(z, "scaled:center") <- NULL # added by QR process
    dimnames(z) <- NULL
    z
  }

  # Claude Sonnet 5 2026-11-21          entire block
  # yde: parses exactly like blrm()'s own ppo formula (see blrm.r's
  # identical block) into a design matrix Z. Predictors in yde may
  # (and usually will) overlap formula's own predictors, per Frank's
  # explicit instruction.
  Z <- zatr <- zsformula <- zTerms <- NULL
  qyde <- 0L
  if (length(yde)) {
    Z <- rms::Design(data2, formula = yde)
    zatrx <- attributes(Z)
    zsformula <- zatrx$sformula
    zTerms <- zatrx$terms
    attr(zTerms, "formula") <- yde
    zatr <- zatrx$Design
    zmmcolnames <- zatr$mmcolnames
    Z <- model.matrix(zTerms, Z)
    zalt <- attr(zmmcolnames, "alt")
    if (!all(zmmcolnames %in% colnames(Z)) && length(zalt)) zmmcolnames <- zalt
    Z <- Z[, zmmcolnames, drop = FALSE]
    colnames(Z) <- zatr$colnames
    qyde <- ncol(Z)
  }

  wqrX <- selectedQr(X, center = TRUE)
  Xs <- trimit(wqrX$X)
  xbar <- wqrX$xbar
  xqrsd <- apply(Xs, 2, sd)
  wqrX$X <- NULL

  zbar <- NULL
  if (qyde > 0) {
    wqrZ <- selectedQr(Z, center = TRUE)
    Zs <- trimit(wqrZ$X)
    zbar <- wqrZ$xbar
    wqrZ$X <- NULL
  }

  n <- nrow(X)
  p <- ncol(X)

  ylev <- ay$levels
  mediany <- ay$median
  midy <- ay$mid
  numy <- ay$freq
  names(numy) <- ylev

  # Claude Sonnet 5 2026-11-16          entire block
  # ytrans's own CENTERING is dropped (ytransmat, the matrix stored in
  # the fit object and used by bormStats()/predict.borm()/bormYdeBT()'s
  # final back-transform, is ytrans(ylev) exactly as the user wrote it,
  # with no centering). Confirmed directly (borm-unequal-var.r example):
  # centering ytrans's own columns by their mean over the observed Y
  # distribution introduces a term linear in Z but constant across Y
  # (Z0 %*% tau0 %*% ycenter) that has no clean home -- Z's OWN
  # centering is absorbed cleanly into alpha_j (which already varies
  # freely by Y-level), but this term does not, and since yde predictors
  # are typically ALSO in the main formula (that is the point of a
  # non-proportional-odds effect), it gets silently absorbed into beta
  # instead -- toggling centering off (holding scaling fixed) left tau
  # essentially unchanged (11.04 -> 11.43) but shifted beta by 6.00,
  # matching the predicted shift tau0*ycenter (5.91) almost exactly, and
  # removing it cost nothing in sampling (divergences statistically
  # unchanged, 38.2% vs 40.0%).
  #
  # SCALING is kept, unlike an earlier version of this fix that dropped
  # it too -- confirmed directly or that choice was NOT free: with no
  # scaling at all, ytrans=function(y) 100*y (on data whose raw y ranges
  # to inv9.5) crashed R outright (segfault/bus error inside Stan's
  # optimizer; ytrans=function(y) 10*y did not), bisected to somewhere
  # between a ytransmat range of ~100 and ~1000. So Stan is given a
  # SEPARATELY scaled ytransmat4stan = ytransmat/yscale (yscale = the SD
  # of ytrans() over the observed, not just distinct, Y values, matching
  # blrm's own cppoy <- cppo(ylev[Y[,1]]) convention), kept numerically
  # safe regardless of ytrans's own units -- while the back-transformed
  # tau (bormYdeBT(), below) is divided by this SAME yscale before
  # being stored, so it remains exactly "per unit of Z, per unit of the
  # user's own ytrans(y)" -- independent of X's, Z's, or Y's sample
  # means, and now also independent of whatever arbitrary scale ytrans
  # itself happens to produce. ytransmat4stan is used ONLY for Stan's
  # own data and for bormYdeBT()'s own alphaadj computation (which must
  # match whatever scale Stan actually used); ytransmat (unscaled) is
  # the one stored/returned and used everywhere else (bormStats(),
  # predict.borm(), bormDoconYde()'s own prior construction aside --
  # see its own call site below, which needs the Stan-matching scale).
  myde <- 0L
  ytransmat <- matrix(0., length(ylev), 0)
  ytransmat4stan <- ytransmat
  yscale <- numeric(0)
  if (qyde > 0) {
    # Claude Sonnet 5 2026-11-22
    # k = length(ylev) - 1 <= 1 (a binary Y, just one cutpoint) makes
    # a Y-dependent effect meaningless -- there is no "shape across
    # cutpoints" to estimate with only one cutpoint, and any such
    # attempt would be exactly redundant with the main formula's own
    # (ordinary, non-yde) effect for that predictor, matching the
    # same non-identifiability rms::orm()'s own yde mechanism has for
    # this case. Per Frank's explicit request; matches the identical
    # check added to orm.fit.r.
    if (length(ylev) <= 2)
      stop("yde/ytrans requires more than 1 modeled intercept (k = ",
           length(ylev) - 1, "); a Y-dependent effect is not meaningful ",
           "for a binary Y")
    # Claude Sonnet 5 2026-11-22          entire block
    # ytrans='all': a special request (character string, not a
    # function) substituted here for a constructed function giving a
    # fully saturated (no-smoothness-assumption) Y-dependent effect,
    # one free tau parameter per cutpoint for intercepts 2..k (the
    # first intercept always gets a zero row, i.e. no y-dependent
    # deviation there) -- k-1 free parameters per yde predictor,
    # matching rms::orm()'s own identical ytrans='all' shortcut
    # (orm.fit.r). The actual column-count formula here (kk-2, using
    # ylev's OWN length) looks offset by one from orm.fit.r's own
    # (kk-1, using ylevels[-1]'s length) -- that is NOT a
    # discrepancy: ylev here is ylevels (all distinct Y values,
    # INCLUDING the first/baseline one with no intercept of its own),
    # one element longer than orm.fit.r's own ylevels[-1], a
    # borm()-vs-orm() convention difference already confirmed and
    # used elsewhere (e.g. Survival.borm()'s own ytransmat[-1, ]).
    # Anchoring "the first intercept" (not "the first element of
    # ylev") is col[e-2] for e>2, which for THIS length-(k+1) ylev is
    # exactly kk-2 columns, numerically equal to k-1.
    if (identical(ytrans, "all")) {
      anycens <- sum(Ncens1) > 0
      if (anycens)
        stop("ytrans='all' does not support censored Y")
      if (length(ylev) - 1 > 50)
        stop("ytrans='all' requires k <= 50 distinct Y values (k = ",
             length(ylev) - 1, ")")
      # Claude Sonnet 5 2026-11-22
      # Cannot write function(y, ylevels = ylev) here the way
      # orm.fit.r's own 'all' block does: borm() always stores ytrans
      # as object$ytrans <- deparse(ytrans) (to avoid serializing a
      # captured environment -- see the re-hydration comment at
      # bormPredictYde()'s own ytrans <- eval(parse(text =
      # object$ytrans))), and a closure-captured default argument
      # (ylev, a variable reference) is NOT part of what deparse()
      # captures -- confirmed directly: re-hydrating such a function
      # from its deparsed text failed with "object 'ylev' not found",
      # since the reconstructed function's default argument still
      # just names a variable, now with nothing bound to it. Building
      # the function from a text string with ylev's own VALUES
      # (not its name) spliced in directly avoids this -- the
      # resulting function's default argument is a literal vector,
      # which deparse() does preserve.
      ytrans <- eval(parse(text = paste0(
        "function(y, ylevels = ", paste(deparse(ylev), collapse = " "), ") {\n",
        "  kk <- length(ylevels)\n",
        "  nn <- length(y)\n",
        "  Z <- matrix(0L, nn, kk - 2L)\n",
        "  colnames(Z) <- as.character(ylevels[3L : kk])\n",
        "  e <- match(y, ylevels)\n",
        "  use <- which(e > 2L)\n",
        "  if (length(use)) Z[cbind(use, e[use] - 2L)] <- 1L\n",
        "  Z\n",
        "}"
      )))
    }
    ytransmat <- ytrans(ylev)
    if (!is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)
    if (NROW(ytransmat) != length(ylev)) {
      stop(
        "ytrans(ylev) must return one row (or value) per element of ylev (",
        length(ylev), "), got ", NROW(ytransmat)
      )
    }
    myde <- ncol(ytransmat)
    if (myde == 0) stop("ytrans must return at least one column")
    yv <- ytrans(ylev[Y[, 1]])
    if (!is.matrix(yv)) yv <- matrix(yv, ncol = 1)
    yscale <- apply(yv, 2, sd)
    yscale[is.na(yscale) | yscale == 0] <- 1
    ytransmat4stan <- sweep(ytransmat, 2, yscale, "/")

    # Claude Sonnet 5 2026-11-16          entire block
    # tausd reinterpreted: a prior SD for tau on a RAW, portable basis
    # -- one unit of the (centered) yde predictor, one SD of
    # ytrans(y) -- not the Stan-internal (QR-rotated Z, yscale-divided
    # ytransmat) basis tau is actually sampled on. yscale (above)
    # already puts ytransmat on a "per SD of ytrans(y)" footing, which
    # is the right, dataset-invariant level to calibrate a default
    # prior against -- confirmed directly, and robustly (not a noisy
    # effect): with the OLD, direct-Stan-internal-scale convention, the
    # SAME tausd implied very different raw-scale prior widths on two
    # datasets with y on scales differing by 100x (and a matching
    # x1000 scale-invariance check on ytrans itself); re-expressing on
    # a "per SD of ytrans(y)" basis removes that dataset-dependence.
    # Only Z's own QR rotation needs undoing here (a pure artifact of
    # that parameterization, scaling as 1/sqrt(n), with no real-world
    # meaning) -- done via Rinv, exactly paralleling bormYdeBT()'s own
    # back-transform, but WITHOUT a yscale divide this time, since
    # yscale is already the right normalization to keep, not undo.
    # NOTE: an earlier version of this comment additionally claimed a
    # specific "tension zone" (a region of degraded NUTS treedepth/
    # leapfrog counts) had been located and could be tuned around --
    # that claim does not hold up: a direct finer-resolution check
    # found leapfrog/treedepth counts chaotically sensitive to tausd
    # (a 0.001% change in tausd swung total leapfrog steps between
    # roughly 200,000 and 350,000, reproducibly) at a resolution far
    # below anything a default-tuning exercise could resolve. The
    # raw/portable-units argument above is independent of that claim
    # and remains correct; only the idea that a specific default value
    # was empirically optimized against sampling efficiency is
    # withdrawn -- tausd's default (below) is a reasonable, round
    # choice on this new basis, not a tuned one.
    # For qyde==1 (every case actually tested), sqrt(rowSums(Rinv^2))
    # reduces exactly to |Rinv|. For qyde>1, Rinv's rows mix predictors
    # together under QR rotation, so there is no single scalar that
    # exactly undoes this per-predictor; using the mean row scale
    # (exact when qyde==1) is a reasonable, but NOT independently
    # validated, generalization -- flagged for Frank's review rather
    # than presented as settled, since every validation run here used
    # qyde==1 (yde's own primary, blrm-cppo-paralleling use case).
    rowscale <- sqrt(rowSums(as.matrix(wqrZ$Rinv)^2))
    tausd <- tausd / mean(rowscale)
  }

  nrp <- length(ylev) - 1L
  kmid <- if (nrp == 1) {
    1L
  } else if (length(mediany)) {
    max(1, which(ylev == mediany) - 1L)
  } else {
    max(1, which((1L:length(ylev)) == mediany) - 1L)
  }

  ass <- rms::DesignAssign(atr, nrp, Terms)
  zass <- if (qyde > 0) rms::DesignAssign(zatr, 0, zTerms)

  # Claude Sonnet 5 2026-11-19          entire block
  # Precompute the per-observation intercept subscripts (ia, ia2, sgn)
  # that drive the log-likelihood, exactly as blrm() now does -- see
  # blrm.r's own identical block for the full rationale. kalpha is the
  # number of intercepts (one less than the number of outcome
  # categories k, which here means the number of DISTINCT Y VALUES,
  # i.e. nrp + 1).
  kalpha <- k - 1L
  a <- Y[, 1]
  b <- Y[, 2]
  y0 <- a - 1L
  y20 <- b - 1L
  y0[is.infinite(a)] <- -1L
  y20[is.infinite(b)] <- kalpha + 1L
  idx <- rms::ormIdx(y0, y20, k = kalpha)

  link <- match(family, c("logistic", "probit", "loglog", "cloglog", "cauchit"))

  d <- list(
    X = Xs, N = n, p = p, k = k, link = link,
    q = qyde, m = myde,
    Z = if (qyde > 0) Zs else matrix(0., n, 0),
    ytransmat = ytransmat4stan, tausd = tausd,
    ia = idx$ia, ia2 = idx$ia2, sgn = idx$sgn,
    conc = conc,
    cn = 0L, C = array(0., c(0, p)), cmus = array(numeric(0)), csds = array(numeric(0)),
    # Claude Sonnet 5 2026-11-24          3 lines
    # These npcontrast defaults were previously only set later (in the
    # pcontrast/npcontrast processing block, after fit.object/mod are
    # built), AFTER the standata=TRUE early return just below -- so
    # that path returned a data list missing cn2/C2/cmus2/csds2
    # entirely, invalid for the actual Stan model whenever npcontrast
    # was not used (confirmed directly: standata=TRUE's own result fed
    # back into rstan::optimizing() errored with "variable does not
    # exist ... variable name=cn2"). Moved up here to match cn/C/
    # cmus/csds's own already-correct placement.
    cn2 = 0L, C2 = array(0., c(0, p + qyde * myde)),
    cmus2 = array(numeric(0)), csds2 = array(numeric(0))
  )

  Nc <- 0L
  d$psigma <- as.integer(psigma)
  if (length(cluster)) {
    d$rsdmean <- as.array(rsdmean)
    d$rsdsd <- if (psigma == 1 || use_mre) as.array(rsdsd) else numeric(0)
    cl <- as.integer(as.factor(cluster))
    Nc <- max(cl, na.rm = TRUE)
    d$Nc <- Nc
    d$cluster <- cl
    d$use_mre <- as.integer(use_mre)
    d$mre <- if (use_mre) as.double(mix_re) else numeric(0)
    method <- "sampling"
  } else {
    Nc <- d$Nc <- 0L
    d$rsdmean <- d$rsdsd <- numeric()
    d$cluster <- integer()
    d$use_mre <- 0L
    d$mre <- numeric()
  }

  if (any(is.na(Xs)) | any(is.na(Y))) stop("program logic error")

  fitter <- "ormcon"
  if (standata) {
    return(d)
  }

  switch(backend,
    rstan = {
      mod <- stanmodels[[fitter]]
      stancode <- rstan::get_stancode(mod)
    },
    cmdstan = {
      sfile <- file.path(
        system.file(package = "rmsb"), "stan",
        paste0(fitter, ".stan")
      )
      stancode <- readLines(sfile)
    }
  )

  hashobj <- list(
    d, pcontrast, inito, inits, iter, warmup, chains, loo, ppairs, method,
    stancode, sampling.args, ...
  )
  datahash <- digest::digest(hashobj)
  if (length(prevhash) && prevhash == datahash) {
    # Claude Sonnet 5 2026-11-21          2 lines
    # Re-hydrate ytrans from its deparsed text, same as blrm.r does for
    # a cached cppo -- a returned cache hit must have a real function
    # there, not the bare string it was stored as.
    if (length(prevfit$ytrans)) prevfit$ytrans <- eval(parse(text = prevfit$ytrans))
    return(prevfit)
  }
  hashobj <- NULL

  itfailed <- function(w) is.list(w) && length(w$fail) && w$fail

  opt <- parm <- NULL

  if (backend == "cmdstan") {
    if (!requireNamespace("cmdstanr", quietly = TRUE)) {
      stop("to use cmdstan backend you must install the cmdstanr package")
    }
    mod <- suppressMessages(cmdstanr::cmdstan_model(sfile, dir = rmsbdir))
  }

  # Claude Sonnet 5 2026-11-17          2 lines
  # pppo/cppo are NOT stored on borm()'s own fit object (unlike blrm(),
  # where they are real, user-facing fields) -- they are blrm()-only
  # concepts that borm() has no use for (borm()'s own non-PO mechanism
  # is qyde/ytrans/tau, entirely different fields). Previously stored
  # here as 0L/NULL placeholders so that blrm()'s shared predict/Mean/
  # Quantile/ExProb methods (which predict.borm()/Mean.borm()/
  # Quantile.borm()/ExProb.borm() delegate to for a non-yde fit) would
  # not crash reading object$pppo with no fallback -- confirmed
  # directly that predict.blrm() does read object$pppo unconditionally
  # with no such fallback. Moved to being injected onto a local,
  # temporary copy of the object at each of those four delegation call
  # sites instead (see bormBlrmDelegate(), below), so a borm() user
  # inspecting their own fit object (names(fit), str(fit)) never sees
  # these blrm-only fields.
  fit.object <- list(
    call = call, fitter = fitter, family = family,
    yname = yname, yunique = ylev, freq = numy, mediany = mediany,
    Design = atr, zDesign = zatr,
    scale.pred = switch(family,
      logistic = c("log odds", "Odds Ratio"),
      loglog   = c("log hazard", "Hazard Ratio"),
      c(family, "")
    ),
    terms = Terms, assign = ass, zassign = zass,
    na.action = atrx$na.action, fail = FALSE,
    non.slopes = nrp, interceptRef = kmid,
    sformula = sformula, zsformula = zsformula,
    pcontrast = pcontrast, npcontrast = npcontrast,
    # Claude Sonnet 5 2026-11-21          3 lines
    # yde: qyde (0 when yde absent) and the (possibly wrapped,
    # center/scale-baked-in) ytrans deparsed as text -- same convention
    # as blrm's own cppo=deparse(cppo), to avoid serializing a huge
    # captured environment; re-hydrated with eval(parse(text=...)) the
    # same way blrm.r re-hydrates a cached cppo (see borm()'s own file
    # caching block above). myde is NOT stored (Claude Sonnet 5
    # 2026-11-23: matches rms::orm()'s own convention -- recomputed as
    # ncol(ytransmat) everywhere it's needed, so there is only one
    # place, ytransmat itself, that can disagree with reality).
    yde = yde, qyde = qyde,
    ytrans = if (qyde > 0) deparse(ytrans), zbar = zbar
  )

  fo <- fit.object

  if (length(pcontrast)) {
    w <- docon(fo, X, wqrX, pcontrast, main = TRUE, ylev = ylev, pppo = 0L,
               clas = c("borm", "rmsb", "rms"))
    fit.object$Contrast <- w$Contrast
    # Claude Sonnet 5 2026-11-22          1 line
    # docon() returns the QR-transformed contrast under the name C,
    # not Contrastt (confirmed directly: w$Contrastt is always NULL,
    # the same pre-existing bug fixed in blrm.r's identical lines).
    fit.object$Contrastt <- w$C
    d$cn <- w$cn
    d$C <- w$C
    d$cmus <- w$cmus
    d$csds <- w$csds
  }

  d$cn2 <- 0L
  d$C2 <- array(0., c(0, p + qyde * myde))
  d$cmus2 <- d$csds2 <- array(numeric(0))
  if (length(npcontrast)) {
    w2 <- bormDoconYde(fo, X, wqrX, zatr, zTerms, zass, zsformula,
                        Z, wqrZ, ytransmat4stan, qyde, myde, npcontrast, ylev)
    fit.object$nContrastt <- w2$C
    d$cn2 <- w2$cn
    d$C2 <- w2$C
    d$cmus2 <- w2$cmus
    d$csds2 <- w2$csds
  }

  ## Need to negate alphas from Stan? No -- borm always uses the
  ## Dirichlet-derived alpha (no iprior=1/2 choice, unlike blrm), so
  ## alphasign is always 1.
  alphasign <- 1.

  if (method != "sampling") {
    d$conc <- 1.0 # see borm()'s own documentation: closer to MLE for optimizing
    if (showopt) {
      g <- switch(backend,
        rstan   = rstan::optimizing(mod, data = d, init = inito, ...),
        cmdstan = mod$optimize(data = d, init = if (!all(inito == "random")) inito, ...)
      )
    } else {
      if (backend == "rstan") {
        capture.output(g <- rstan::optimizing(mod, data = d, init = inito, ...))
      } else {
        capture.output(g <- mod$optimize(
          data = d,
          init = if (!all(inito == "random")) inito, ...
        ))
      }
    }
    d$conc <- conc # restore
    rc <- switch(backend,
      rstan = g$return_code,
      cmdstan = g$return_codes()
    )
    if (rc != 0) {
      warning(paste(
        "Optimizing did not work; return code", rc,
        "\nPosterior modes not computed"
      ))
      opt <- list(
        coefficients = NULL,
        return_code = rc, hessian = NULL
      )
    } else {
      parm <- switch(backend,
        rstan = g$par,
        cmdstan = g$mle()
      )
      nam <- names(parm)
      al <- nam[grep("alpha\\[", nam)]
      be <- nam[grep("beta\\[", nam)]
      alphas <- alphasign * parm[al]
      betas <- parm[be]
      betas <- matrix(betas, nrow = 1) %*% t(wqrX$Rinv)
      names(betas) <- atr$colnames
      alphas <- alphas - sum(betas * xbar)
      taus <- NULL
      if (qyde > 0) {
        ta <- nam[grep("tau\\[", nam)]
        bt <- bormYdeBT(parm[ta], qyde, myde, wqrZ$Rinv, zbar, ytransmat4stan, yscale)
        alphas <- alphas - bt$alphaadj
        taus <- as.vector(bt$tau)
        names(taus) <- bormYdeNames(colnames(Z), ytransmat, qyde, myde)
      }
      names(alphas) <- if (nrp == 1) "Intercept" else paste0("y>=", ylev[-1])
      opt <- list(
        coefficients = c(alphas, betas, taus),
        deviance = -2 * switch(backend,
          rstan = g$value,
          cmdstan = g$lp()
        ),
        return_code = g$return_code, hessian = g$hessian
      )
    }
    if (method == "optimizing") {
      return(opt)
    }
  }

  if (progress != "") sink(progress, append = TRUE)

  debug(1)
  sigmaname <- if (use_mre) c("sigma1", "sigma2") else "sigmag"
  incpars <- c("alpha", "beta", if (qyde > 0) "tau", if (length(cluster)) c("gamma", sigmaname),
               if (backend == "rstan" & loo) "log_lik")

  stime <- system.time({
    args <- switch(backend,
      # Claude Sonnet 5 2026-11-24          1 line
      # init=inits was previously omitted entirely from the rstan
      # branch's args (only the cmdstan branch set it), so any user-
      # supplied inits -- e.g. seeding sampling from the optimizer's
      # own solution -- was silently ignored for backend='rstan'.
      # rstan::sampling() accepts the literal string "random" (inits'
      # own default) directly, same as inito already is passed
      # unconditionally to rstan::optimizing() above, so no
      # conditional wrapper is needed here.
      rstan = c(
        list(mod,
          pars = incpars, data = d, iter = iter,
          warmup = warmup, chains = chains,
          refresh = refresh, init = inits
        ),
        sampling.args
      ),
      cmdstan = c(
        list(
          data = d, iter_sampling = iter - warmup,
          iter_warmup = warmup, chains = chains,
          refresh = refresh,
          init = if (!all(inits == "random")) inits
        ),
        sampling.args
      )
    )
    g <- switch(backend,
      rstan   = do.call(rstan::sampling, args),
      cmdstan = do.call(mod$sample, args)
    )
  })
  sampling_time <- unname(stime["elapsed"])
  if (backend == "rstan" && (is.null(g) || g@mode == 2)) {
    stop("Stan rstan sampler failed")
  }
  debug(2)

  if (progress != "") sink()

  draws <- switch(backend,
    cmdstan = {
      draws <- g$draws()
      nam <- dimnames(draws)[[3]]
      regx <- paste(paste0("^", incpars, "\\["), collapse = "|")
      nam <- nam[grep(regx, nam)]
      matrix(draws[, , nam], ncol = length(nam), dimnames = list(NULL, nam))
    },
    rstan = {
      nam <- names(g)
      as.matrix(g)
    }
  )

  al <- nam[grep("alpha\\[", nam)]
  be <- nam[grep("beta\\[", nam)]
  ga <- nam[grep("gamma\\[", nam)]

  ndraws <- nrow(draws)
  alphas <- alphasign * draws[, al, drop = FALSE]
  betas <- draws[, be, drop = FALSE]
  betas <- betas %*% t(wqrX$Rinv)

  omega <- NULL
  clparm <- character(0)
  gammas <- NULL

  if (length(cluster)) {
    omeganame <- if (use_mre) c("sigma1[1]", "sigma2[1]") else "sigmag[1]"
    omega <- draws[, omeganame, drop = FALSE]
    colnames(omega) <- if (use_mre) c("sigma1", "sigma2") else "sigmag"
    clparm <- colnames(omega)
    cle <- draws[, ga, drop = FALSE]
    gammas <- apply(cle, 2, median) # posterior median per subject
  }

  debug(3)

  # Claude Sonnet 5 2026-11-21          entire block
  # yde: back-transform tau draw-by-draw (bormYdeBT()), via the same
  # q x m / k x m generalization of blrm's own constrained-PPO tau
  # back-transform documented in bormYdeBT()'s own roxygen. A loop
  # over draws is the simplest correct approach (qyde, myde are small;
  # ndraws is at most a few times chains*iter) -- matching blrm's own
  # per-column loop for its unconstrained PPO tau (see blrm.r's mtaus
  # block) in spirit, generalized here to loop over draws instead of
  # over columns since BOTH the transform (Rinv) and the resulting
  # alpha adjustment vary by draw here, unlike blrm's single point
  # estimate case.
  taus <- NULL
  tauadj <- matrix(0., ndraws, nrp)  # per-draw, per-alpha-column; stays 0 when qyde==0
  if (qyde > 0) {
    ta <- nam[grep("tau\\[", nam)]
    clparm <- c(clparm, ta)
    tauraw <- draws[, ta, drop = FALSE]
    taus <- matrix(0., ndraws, qyde * myde)
    for (i in 1:ndraws) {
      bt <- bormYdeBT(tauraw[i, ], qyde, myde, wqrZ$Rinv, zbar, ytransmat4stan, yscale)
      taus[i, ] <- as.vector(bt$tau)
      tauadj[i, ] <- bt$alphaadj
    }
    colnames(taus) <- bormYdeNames(colnames(Z), ytransmat, qyde, myde)
  }

  # alphacorr (length ndraws) is the SAME shift for every alpha column
  # (X's centering doesn't depend on Y-level); tauadj (ndraws x nrp)
  # varies by alpha column (Y-level-dependent), so it is subtracted as
  # a full matrix, not recycled like alphacorr.
  alphacorr <- rowSums(sweep(betas, 2, xbar, "*"))
  alphas <- sweep(alphas, 1, alphacorr, "-") - tauadj

  colnames(alphas) <- if (nrp == 1) "Intercept" else paste0("y>=", ylev[-1])
  colnames(betas) <- atr$colnames

  draws <- cbind(alphas, betas, taus)

  debug(4)
  param <- rbind(mean = colMeans(draws), median = apply(draws, 2, median))
  if (method != "sampling") {
    param <- rbind(mode = opt$coefficients, param)
    opt$coefficients <- NULL
  }

  Loo <- lootime <- NULL
  if (loo) {
    lootime <- system.time(
      Loo <- tryCatch(
        switch(backend,
          rstan = rstan::loo(g, cores = cores_loo),
          cmdstan = g$loo(cores = cores_loo)
        ),
        # Claude Sonnet 5 2026-11-23          1 line
        # Capture the real error message instead of discarding it --
        # confirmed directly that loo() throws a real, specific error
        # (e.g. "All input values must be finite or -Inf", "NAs not
        # allowed in input") whenever log_lik contains even one -Inf/
        # NaN entry, which this package's own "loo failed" message
        # previously gave no way to see.
        error = function(e) list(fail = TRUE, message = conditionMessage(e))
      )
    )
    if (itfailed(Loo)) {
      warning("loo failed: ", Loo$message,
              "; try running loo(stanGet(fit object)) for more information")
      Loo <- NULL
    }
  }

  debug(5)
  switch(backend,
    rstan = {
      diagnostics <-
        tryCatch(rstan::summary(g, pars = c(al, be, clparm), probs = NULL),
          error = function(...) list(fail = TRUE)
        )
      if (itfailed(diagnostics)) {
        warning("rstan::summary failed; see fit component diagnostics")
        diagnostics <- list(pars = c(al, be, clparm), failed = TRUE)
      } else {
        diagnostics <- diagnostics$summary[, c("n_eff", "Rhat")]
      }
    },
    cmdstan = {
      vsum <- setdiff(incpars, "gamma")
      dm <- capture.output(g$cmdstan_diagnose())[-(1:2)]
      diagnostics <- list(
        message = dm,
        diagnostic_summary = g$diagnostic_summary(),
        summary = as.data.frame(g$summary(variables = vsum))
      )
    }
  )

  debug(6)

  if (length(ppairs)) {
    if (backend == "cmdstan") {
      if (!requireNamespace("bayesplot", quietly = TRUE)) {
        stop("bayesplot package must be installed if using ppairs and cmdstan")
      }
    }
    if (is.character(ppairs)) png(ppairs, width = 1000, height = 1000, pointsize = 11)
    ppa <- tryCatch(
      switch(backend,
        rstan = pairs(g, pars = c(al[1], be, clparm)),
        cmdstan = print(bayesplot::mcmc_pairs(g$draws(c(al[1], be, clparm)),
          np = bayesplot::nuts_params(g)
        ))
      ),
      error = function(...) list(fail = TRUE)
    )
    if (itfailed(ppa)) warning("ppairs failed, probably because of too many parameters for available space")
    if (itfailed(ppa) || is.character(ppairs)) dev.off()
  }

  debug(7)

  res <- c(
    fit.object,
    list(
      draws = draws, omega = omega, gammas = gammas,
      param = param,
      psigma = psigma, rsdmean = rsdmean, rsdsd = rsdsd,
      xqrsd = xqrsd,
      N = n, Ncens1 = Ncens1, Ncens2 = Ncens2, p = p,
      alphas = al, betas = be,
      xbar = xbar,
      # Claude Sonnet 5 2026-11-21          1 line
      # wqrZ$Rinv (only, not the whole QR object) is needed later by
      # yde-aware prediction; NULL (no duplicate entry) when qyde==0.
      wqrZRinv = if (qyde > 0) wqrZ$Rinv,
      x = if (x) X, y = if (y) Y, z = if (x && qyde > 0) Z,
      loo = Loo, lootime = lootime,
      clusterInfo = if (length(cluster)) {
        list(
          cluster = if (x) cluster else NULL,
          n = Nc, name = clustername,
          mix_re = if (use_mre && x) mix_re else NULL,
          mix_rename = mix_rename
        )
      },
      opt = opt, diagnostics = diagnostics,
      iter = iter, chains = chains, stancode = stancode, datahash = datahash,
      backend = backend, sampling_time = sampling_time
    )
  )

  res$conc <- conc
  class(res) <- c("borm", "rmsb", "rms")
  if (length(file)) {
    saveRDS(res, file, compress = "xz")
  }
  res$rstan <- g
  res
}

# Claude Sonnet 5 2026-11-21          entire block
# Internal borm() helpers for the yde/ytrans mechanism, used by
# both the optimizing-mode (single point estimate) and sampling-mode
# (full posterior draws) branches of borm() itself -- top-level per
# this package's standing rule that a helper used more than once must
# not live inside another function's body.

##' Back-Transform a borm() yde tau Vector/Draw
##'
##' Internal helper for [borm()]: back-transforms one posterior draw's (or the optimizing-mode point estimate's) flattened `tau` coefficient vector from the QR-transformed `Z` scale to the original covariate scale, and computes the intercept (`alpha`) adjustment implied by `Z`'s own centering -- the direct generalization, to a `qyde x myde` `tau` matrix and a `k x myde` `ytransmat`, of [blrm()]'s own constrained-PPO `tau` back-transform (`taus <- wqrZ$Rinv %*% taus; alphas <- alphas - d$pposcore[-1] * sum(taus * zbar)`, a `q x 1` `tau` and a length-`k` `pposcore`).
##' @param tauvec a length `qyde*myde` vector (one posterior draw, or the point estimate), in Stan's own column-major matrix-parameter order (`tau[1,1], tau[2,1], ..., tau[qyde,1], tau[1,2], ...`)
##' @param qyde number of `yde` predictors
##' @param myde number of `ytrans` columns
##' @param Rinv the `qyde x qyde` inverse QR matrix from `selectedQr(Z)`
##' @param zbar the length-`qyde` vector of `Z`'s original-scale column means from `selectedQr(Z)`
##' @param ytransmat the `k x myde` matrix actually passed to Stan (i.e. `ytransmat4stan`, scaled by `yscale`, NOT the unscaled matrix `borm()` stores/returns)
##' @param yscale the length-`myde` scale vector `ytransmat4stan` was divided by (`ytransmat/yscale == ytransmat4stan`); `tau` is multiplied back by it so the returned value is on the user's own, unscaled `ytrans(y)` units
##' @return a list with `tau` (the `qyde x myde` back-transformed coefficient matrix, on `Z`'s original scale and the user's own unscaled `ytrans(y)` units) and `alphaadj` (a length-`(k-1)` vector to subtract from `alpha`, one entry per alpha index)
##' @author Frank Harrell
##' @noRd
bormYdeBT <- function(tauvec, qyde, myde, Rinv, zbar, ytransmat, yscale) {
  Tm  <- matrix(tauvec, nrow = qyde, ncol = myde)
  Tbt <- Rinv %*% Tm
  zbartau <- as.vector(matrix(zbar, nrow = 1) %*% Tbt)
  alphaadj <- as.vector(ytransmat[-1, , drop = FALSE] %*% zbartau)
  # Claude Sonnet 5 2026-11-16          2 lines
  # Scale back to the user's own ytrans(y) units AFTER alphaadj above,
  # which must match Stan's own (scaled) ytransmat, not the raw one.
  tau0 <- sweep(Tbt, 2, yscale, "/")
  list(tau = tau0, alphaadj = alphaadj)
}

##' Column Labels for a borm() ytransmat
##'
##' Internal helper for [borm()]: labels for `tau`'s columns, using `ytransmat`'s own column names when the user's `ytrans` function supplied them (and they are unique), else generic `yt1, yt2, ...` labels (just `yt`, no number, when there is only one column).
##' @param ytransmat the `k x myde` `ytrans(ylev)` matrix
##' @param myde number of `ytrans` columns
##' @return a length-`myde` character vector
##' @author Frank Harrell
##' @noRd
bormYdeColLab <- function(ytransmat, myde) {
  cn <- colnames(ytransmat)
  # Claude Sonnet 5 2026-11-21          6 lines
  # Reject names cbind() auto-infers from a bare-symbol argument (e.g.
  # ytrans <- function(y) cbind(y, y^2) gives colnames c("y", "") --
  # confirmed directly), not a deliberate user label; an empty name
  # (or no names at all) is diagnostic of this and falls back to
  # generic yt1,yt2,... (or just "yt" for a single column). A fully-
  # named set IS trusted as deliberate, but must be unique -- an error,
  # not a silent fallback, since a duplicate is almost certainly a
  # mistake the user would want to know about.
  if(!length(cn) || any(cn == "")) cn <- if(myde == 1L) "yt" else paste0("yt", 1 : myde)
  else if(anyDuplicated(cn)) stop("ytrans's column names must be unique")
  cn
}

##' Flattened Names for a borm() yde tau Matrix
##'
##' Internal helper for [borm()]: names for `tau`'s flattened (column-major, matching `as.vector()`/Stan's own convention) entries, as `"<Z column> * <ytrans column>"` -- Claude Sonnet 5 2026-11-23: separator changed from `" x "` to `" * "` to match `rms::orm.fit()`'s own `tauname` convention (`outer(zname, ytransname, paste, sep=" * ")`) exactly, a user-visible string previously inconsistent between the two packages.
##' @param Zcolnames column names of the `yde` design matrix `Z`
##' @param ytransmat the `k x myde` `ytrans(ylev)` matrix
##' @param qyde number of `yde` predictors
##' @param myde number of `ytrans` columns
##' @return a length-`qyde*myde` character vector
##' @author Frank Harrell
##' @noRd
bormYdeNames <- function(Zcolnames, ytransmat, qyde, myde) {
  paste(rep(Zcolnames, myde), rep(bormYdeColLab(ytransmat, myde), each = qyde), sep = " * ")
}

##' Combined beta+tau Contrast Row Builder for borm()'s npcontrast
##'
##' Internal helper for [borm()]: generalizes `docon()`'s single-design-matrix contrast mechanism so that `npcontrast`'s named settings can produce a row spanning BOTH `beta`'s `p` columns and `tau`'s flattened `qyde*myde` columns, letting contrast expressions reference betas, taus, or both -- needed because `beta` and `tau` are mechanically separate Stan parameter blocks, so a single-design-matrix row (as `docon()` builds for `pcontrast`, or as `blrm()`'s own single-column-`tau` `npcontrast` uses) cannot express a quantity like "the effect of a predictor at a given Y value", which is `beta_j + Z_j . tau . ytransmat[y,]` -- necessarily mixing both. `ycut` is read from `pc$ycut` exactly as `docon()` reads it for `blrm()`'s own `cppo`-weighted `npcontrast` (a single scalar applied to every named setting, or a named vector giving a different Y cutoff per named setting); a setting with no `ycut` (absent entirely, or named but missing that particular setting) gets an all-zero tau-part, i.e. a pure-`beta` row -- and symmetrically, two settings sharing the same covariate values but different `ycut` have their `beta`-parts cancel exactly in a difference, giving a pure-`tau` contrast; this is confirmed, not just reasoned through (see `inst/tests/borm-yde.r`'s own npcontrast tests). Each setting's row is QR-transformed individually (`beta`-part via `wqrX$Rinv`, `tau`-part via `wqrZ$Rinv` applied BEFORE the outer-product expansion against `ytransmat`, since `ytransmat`'s row varies per setting via `ycut` and so cannot be factored into one whole-matrix transform applied once at the end, unlike `docon()`'s own single-design-matrix case) -- since matrix right-multiplication is linear, this is mathematically identical to transforming the combined contrast matrix once at the end (confirmed directly against `docon()`'s own existing single-column `ycut`-weighted formula, which this exactly reduces to when `myde=1`), so no further whole-matrix QR-transform step is needed afterward.
##' @param fo a partially-built fit object (as `docon()` itself builds)
##' @param X the main-formula design matrix
##' @param wqrX the QR decomposition of `X`
##' @param zatr,zTerms,zass,zsformula the `yde` formula's own `Design`/`terms`/`assign`/`sformula`, as computed in [borm()] when building `Z`
##' @param Z the `yde` design matrix
##' @param wqrZ the QR decomposition of `Z`
##' @param ytransmat the `k x myde` `ytrans(ylev)` matrix
##' @param qyde,myde dimensions of `tau`
##' @param pc the user-supplied `npcontrast` list
##' @param ylev vector of distinct ordered Y values/levels
##' @return a list with `cn`, `C` (already QR-transformed, `p+qyde*myde` columns wide), `cmus`, `csds`
##' @author Frank Harrell
##' @noRd
bormDoconYde <- function(fo, X, wqrX, zatr, zTerms, zass, zsformula,
                          Z, wqrZ, ytransmat, qyde, myde, pc, ylev) {
  nint <- length(ylev) - 1
  p <- ncol(X)
  fo$coefficients <- rep(0., nint + p)
  fo$tau <- numeric(0)
  fo$draws <- 1L # trick predictrms into bayes=TRUE
  fo$param <- rbind(mean = rep(0., nint + p), median = rep(0., nint + p))
  class(fo) <- c("borm", "rmsb", "rms")
  # fo is a stub for Xcontrast()'s own internal predict() call, not a
  # complete fit -- see docon()'s identical fix (stanMisc.r) for why
  # qyde must be forced to 0 here, or predict.borm() misroutes to
  # bormPredictYde() on this incomplete object.
  fo$qyde <- 0L

  foX <- fo
  foZ <- fo
  foZ$Design   <- zatr
  foZ$terms    <- zTerms
  foZ$assign   <- zass
  foZ$sformula <- zsformula

  weights <- pc$weights
  if (!length(weights)) weights <- "equal"
  expand <- pc$expand
  if (!length(expand)) expand <- TRUE
  cmus <- pc$mu
  csds <- pc$sd
  con <- pc$contrast
  if (!length(con)) stop("must include contrast in npcontrast")
  if (!is.expression(con)) stop("contrast must be expression()")
  ycut <- pc$ycut
  pc[c("weights", "expand", "mu", "sd", "contrast", "ycut")] <- NULL
  cnam <- names(pc)
  if (!length(cnam) || any(cnam == "")) {
    stop("predictor setting lists in npcontrast must be named")
  }
  ynamed <- length(names(ycut)) > 0

  XC <- Map(
    function(x, nm) {
      xrow <- do.call(
        Xcontrast,
        list(foX, a = x, weights = weights, expand = expand, Zmatrix = FALSE)
      )
      xrow_qr <- xrow %*% wqrX$Rinv
      yc <- if (ynamed) {
        if (nm %in% names(ycut)) ycut[[nm]] else NA
      } else if (length(ycut)) {
        ycut[1]
      } else {
        NA
      }
      if (!is.na(yc)) {
        zrow <- do.call(
          Xcontrast,
          list(foZ, a = x, weights = weights, expand = expand, Zmatrix = FALSE)
        )
        zrow_qr <- zrow %*% wqrZ$Rinv
        yidx <- if (all.is.numeric(ylev)) {
          which(ylev == yc)
        } else {
          max((1:length(ylev))[ylev <= yc])
        }
        zexp_qr <- matrix(outer(as.vector(zrow_qr), ytransmat[yidx, ]), nrow = 1)
      } else {
        zexp_qr <- matrix(0., 1, qyde * myde)
      }
      cbind(xrow_qr, zexp_qr)
    },
    pc, cnam
  )

  Contrastt <- do.call(rbind, lapply(con, eval, XC))
  cn <- nrow(Contrastt)
  if (!length(cmus)) cmus <- 0.
  if (!length(csds)) csds <- 100.
  if (length(cmus) == 1) cmus <- rep(cmus, length = cn)
  if (length(csds) == 1) csds <- rep(csds, length = cn)
  if (length(cmus) != cn || length(csds) != cn) {
    stop(
      "mismatch between number of contrasts (", cn,
      "), length of mu (", length(cmus),
      "), and length of sd (", length(csds), ")"
    )
  }
  list(cn = cn, C = as.array(Contrastt), cmus = as.array(cmus), csds = as.array(csds))
}

##' Resolve newdata for a borm() Prediction Call
##'
##' Internal helper for [predict.borm()]/`bormPredictYde()`: finds the data frame to predict at, trying, in order, a named `newdata=` argument, a `predict(fit, d, type=...)`-style POSITIONALLY-passed (unnamed) data frame argument -- the convention [rms::contrast.rms()]'s own internal `pred()` wrapper uses -- and finally [rms::gendata()] built from named covariate-value arguments, matching [rms::Xcontrast()]'s own convention. Confirmed necessary directly: `contrast()` on a `yde`-fitted `borm()` object failed with a `gendata()` error before this fix, since `contrast.rms()` passes its data frame positionally, which a plain `dots$newdata` lookup does not see.
##' @param object a [borm()] fit object
##' @param dots the result of `list(...)` from the calling predict method
##' @return a data frame
##' @author Frank Harrell
##' @noRd
bormGetNewdata <- function(object, dots) {
  nd <- dots$newdata
  if (!length(nd)) {
    nms <- names(dots)
    unnamed <- if (length(nms)) which(nms == "") else seq_along(dots)
    dfidx <- unnamed[sapply(unnamed, function(i) is.data.frame(dots[[i]]))]
    if (length(dfidx)) nd <- dots[[dfidx[1]]]
  }
  if (!length(nd)) nd <- do.call(rms::gendata, c(list(object), dots))
  nd
}

##' Error If a borm() Fit Used yde/ytrans
##'
##' Internal helper for [Mean.borm()]/[Quantile.borm()]/[ExProb.borm()]: these delegate entirely to their `blrm()` counterparts, whose logic is hardcoded to a single-column `cppo` and would silently ignore a `yde`/`ytrans` Y-dependent effect rather than erroring -- this stops that from happening silently, until these three functions get their own `yde`-aware implementations (as [predict.borm()] already does for `type='lp'`/`'x'`, via `bormPredictYde()`).
##' @param object a [borm()] fit object
##' @param fname the name of the calling function, for the error message
##' @return nothing; stops with an error if `object$qyde > 0`
##' @author Frank Harrell
##' @noRd
bormStopIfYde <- function(object, fname) {
  qyde <- object$qyde
  if (length(qyde) && qyde > 0) {
    stop(
      fname, " is not yet implemented for a borm() fit using yde/ytrans; ",
      "use predict(object, type='lp') instead"
    )
  }
}

##' Temporarily Attach blrm()-Only Placeholder Fields for Delegation
##'
##' Internal helper for [predict.borm()]/[Mean.borm()]/[Quantile.borm()]/[ExProb.borm()]: `borm()` does not store `pppo`/`cppo` on its own fit object (they are `blrm()`-only concepts `borm()` has no use for -- its own non-PO mechanism is `qyde`/`ytrans`/`tau`, entirely different fields), but the `blrm()` methods these four functions delegate to for a non-`yde` fit read `object$pppo` unconditionally, with no fallback for it being absent (confirmed directly: [predict.blrm()] does `pppo <- object$pppo` then `if (pppo == 0) ...` with no `length()` guard, which errors on `NULL`). Attaches `pppo = 0L`, `cppo = NULL` to a local copy of `object` for the duration of the delegated call only, leaving the object the user actually holds untouched.
##' @param object a [borm()] fit object
##' @return `object`, with `pppo = 0L`, `cppo = NULL`, `ylevels`, and `link` added
##' @author Frank Harrell
##' @noRd
bormBlrmDelegate <- function(object) {
  object$pppo <- 0L
  object$cppo <- NULL
  # Claude Sonnet 5 2026-11-23
  # borm()'s own fields are yunique/family (matching rms::orm()'s
  # convention), but blrm.r's own generic code (blrmStats/print.blrm/
  # Mean.blrm/Quantile.blrm/ExProb.blrm/tauFetch) reads
  # object$ylevels/object$link unconditionally -- bridge both here for
  # the duration of the delegated call, exactly the same pattern as
  # pppo/cppo above. blrm() itself always sets link to a hardcoded
  # "logistic"; here it is just an alias for borm()'s own family.
  object$ylevels <- object$yunique
  object$link    <- object$family
  object
}

##' Build the `yde` Design Matrix `Z` from `newdata`
##'
##' Internal helper, factored out of `bormPredictYde()` (where this logic
##' originally lived inline) so `Survival.borm()` -- which also needs `Z`
##' for its own per-posterior-draw loop -- doesn't duplicate it. Building
##' `Z` at predict time needs two fixes to `rms::Design()`, neither about
##' `yde`/`ytrans` logic itself: (1) a data frame containing columns
##' beyond those `yde`'s own formula references (e.g. the main formula's
##' own `x` alongside `yde`'s `x2`) makes `rms::Design()` fail ("program
##' logic error tl") once more than one `newdata` row is requested,
##' apparently via an interaction with the active global `datadist`
##' registering both variables; fixed by subsetting to only `yde`'s own
##' variables (`all.vars(object$yde)`) first. (2) separately,
##' `rms::Design()` cannot build contrasts for a categorical predictor
##' from `newdata` containing only one of its levels (confirmed: "x2 has
##' <2 category levels" on a plain 1-row data frame), unlike
##' `predictrms()`'s own machinery for the main formula, which reuses
##' `object$Design`'s stored level information instead of re-deriving it
##' from `newdata`; fixed by explicitly releveling factor/character
##' columns to the full level set seen when `yde` was fit (`zatr$values`).
##' @param object a `yde`-fitted [borm()] object
##' @param newdata the resolved covariate data frame (e.g. from
##'   `bormGetNewdata()`)
##' @return the `n x qyde` `yde` design matrix
##' @author Frank Harrell
##' @noRd
bormYdeZ <- function(object, newdata) {
  zatr <- object$zDesign
  nd2 <- newdata
  zvars <- all.vars(object$yde)
  nd2 <- nd2[, zvars, drop = FALSE]
  zvalues <- zatr$values
  for (vn in intersect(names(zvalues), names(nd2))) {
    if (is.character(nd2[[vn]]) || is.factor(nd2[[vn]])) {
      nd2[[vn]] <- factor(nd2[[vn]], levels = zvalues[[vn]])
    }
  }
  Zdes   <- rms::Design(nd2, formula = object$yde)
  zatrx  <- attributes(Zdes)
  zTerms <- zatrx$terms
  attr(zTerms, "formula") <- object$yde
  zmmcolnames <- zatr$mmcolnames
  Z <- model.matrix(zTerms, Zdes)
  zalt <- attr(zmmcolnames, "alt")
  if (!all(zmmcolnames %in% colnames(Z)) && length(zalt)) zmmcolnames <- zalt
  Z <- Z[, zmmcolnames, drop = FALSE]
  colnames(Z) <- zatr$colnames
  Z
}

##' Y-Dependent-Effect-Aware Prediction for a `borm()` Fit Using `yde`/`ytrans`
##'
##' Internal helper for [predict.borm()]: implements `type='lp'` (the posterior mean/median point-estimate case, without `cint` or `fun=`) and `type='x'`, generalizing [predict.blrm()]'s own single-column `cppo` logic (`object$cppo`, a scalar `Z*cppo(ycut)` term) to `borm()`'s `q x m` `tau` matrix and `k x m` `ytransmat`: the Y-dependent design contribution for one observation at one Y cutoff is the `q*m`-length vector `as.vector(outer(Z[row,], ytransmat[cutoff row,]))` (in the same column-major order `tau`'s flattened draws/back-transform use -- see `bormYdeBT()`'s own documentation), dotted against the flattened `tau` coefficients. `newdata` is resolved via [rms::gendata()] exactly as [rms::Xcontrast()] does, since [rms::predictrms()]'s own `second=TRUE` mechanism cannot be reused here (it is keyed off `object$pppo`, a `blrm()`-only field `borm()` does not store on its own fit object -- see `bormBlrmDelegate()` for how `predict.borm()`'s own, separate non-`yde` branch satisfies `predict.blrm()`'s need for it without storing it persistently).
##' @param Zmatrix for `type='x'`, set to `FALSE` to exclude the Y-dependent (`yde`) columns and return only the ordinary design matrix -- matching [predict.blrm()]'s own `Zmatrix` argument, needed so [rms::Xcontrast()]'s `Zmatrix=FALSE` option (used by its own partial-PO contrast machinery) and [rms::Predict()]'s analogous suppression work correctly for a `yde`-fitted `borm()` object too, not just `blrm()`.
##' @param zcppo accepted for interface compatibility with [predict.blrm()] only; has no effect here. `blrm()`'s single-column `cppo` has a meaningful "unweighted" form (the raw `Z` columns, vs. `Z*cppo(ycut)`) that `zcppo` toggles between; `borm()`'s `tau` is inherently a `q x m` matrix matching only the fully-expanded, Y-level-weighted cross-product (`Zexp`, what `Zmatrix=TRUE` returns) -- there is no analogous "unweighted" alternative representation to toggle to.
##' @author Frank Harrell
##' @noRd
bormPredictYde <- function(object, ..., Zmatrix = TRUE, zcppo = TRUE,
                            kint, ycut, fun, funint, type,
                            se.fit, codes, posterior.summary, cint) {
  # Claude Sonnet 5 2026-11-21          6 lines
  # 'fitted'/'fitted.ind'/'mean' ported from rms::ormPredictYde()'s own
  # bilinear factorization (Z %*% Tau %*% t(ytransmat)), generalized to
  # loop once per posterior draw -- the one new dimension orm() never
  # had. Per the explicit modularization decision (not calling rms's
  # unexported internals across the package boundary), this is a
  # parallel implementation, not a shared call.
  implemented <- c("lp", "x", "fitted", "fitted.ind", "mean")
  if (type %nin% implemented) {
    stop(
      "type='", type, "' is not yet implemented for a borm() fit using yde/ytrans; ",
      "only ", paste(sQuote(implemented), collapse=", "), " are currently supported for such fits"
    )
  }
  if (length(fun)) stop("fun= is not yet implemented for a borm() fit using yde/ytrans")
  if (se.fit) warning("se.fit does not apply to Bayesian models")

  dots <- list(...)
  nd <- bormGetNewdata(object, dots)

  X <- rms::predictrms(object, newdata = nd, type = "x")
  n <- nrow(X)
  # Claude Sonnet 5 2026-11-17          2 lines
  # Zmatrix=FALSE: matching predict.blrm()'s own early exit, skip the
  # (expensive) yde/ytrans design-matrix construction below entirely.
  if (type == "x" && !Zmatrix) return(X)

  Z <- bormYdeZ(object, nd)

  qyde <- object$qyde
  # ytrans is stored deparsed (object$ytrans = deparse(ytrans) in
  # borm() itself), the same convention blrm.r uses for cppo, to avoid
  # serializing a captured environment -- EXCEPT on a file= cache hit,
  # where borm() itself already re-hydrates object$ytrans into a real
  # function before returning (so a cache-hit object's own predict()
  # call doesn't have to). Fixed a pre-existing bug, found while
  # testing ytrans='all' but not specific to it (confirmed directly
  # with an ordinary custom ytrans too): unconditionally re-parsing
  # object$ytrans as text failed with "cannot coerce type 'closure' to
  # vector of type 'character'" whenever object was a cache-hit fit,
  # since object$ytrans was already a function there, not text.
  ytrans <- if (is.function(object$ytrans)) object$ytrans
            else eval(parse(text = object$ytrans))
  ylevels <- object$yunique
  ytransmat <- ytrans(ylevels)
  # Claude Sonnet 5 2026-11-16          2 lines
  # ytrans is stored raw (no more auto-centering/scaling wrapper that
  # used to guarantee a matrix return) -- wrap a plain-vector result,
  # exactly as borm()'s own construction of ytransmat already does.
  if (!is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)
  # Claude Sonnet 5 2026-11-23
  # myde is not stored on the object (matches rms::orm()'s own
  # convention) -- always recomputed from the matrix actually built.
  myde <- ncol(ytransmat)

  iref <- object$interceptRef
  if (!length(kint)) kint <- iref
  ycutgiven <- length(ycut) > 0
  if (!ycutgiven) ycut <- ylevels[kint + 1]
  if (ycutgiven) {
    if (type == "lp" && length(ycut) > 1) stop('ycut may only be a scalar for type="lp"')
    # Claude Sonnet 5 2026-11-21          2 lines
    # kint must be re-derived from ycut when ycut is given, exactly as
    # predict.blrm() does -- missed on the first pass (confirmed
    # directly: without this, the alpha/intercept part of lp stayed at
    # the default reference intercept regardless of ycut, while the
    # tau/ytransmat part correctly used ycut, giving a wrong lp).
    kint <- if (all.is.numeric(ylevels)) {
      which(ylevels == ycut[1]) - 1
    } else {
      max((1:length(ylevels))[ylevels <= ycut[1]]) - 1
    }
  }

  if (length(ycut) %nin% c(1, n)) {
    stop("ycut must be of length 1 or the number of requested predictions")
  }
  ycut <- rep(ycut, length = n)
  rowidx <- if (all.is.numeric(ylevels)) {
    match(ycut, ylevels)
  } else {
    sapply(ycut, function(yy) max((1:length(ylevels))[ylevels <= yy]))
  }

  expandZ <- function(Zrow, ytransrow) as.vector(outer(Zrow, ytransrow))
  taunam <- bormYdeNames(colnames(Z), ytransmat, qyde, myde)

  if (type %in% c("fitted", "fitted.ind", "mean")) {
    # Claude Sonnet 5 2026-11-21          entire block
    # All three need the Y-dependent contribution at every cutpoint
    # j=1..k, not just the one given by kint/ycut (unlike 'lp'/'x'
    # above), so kint/ycut/rowidx (computed above for those two) are
    # simply unused here. Loops once per posterior draw -- unlike
    # rms::ormYdeFittedMean()'s single MLE vector, there is no single
    # "the" tau/alpha/beta to use. Each draw's own n x k matrix is
    # built the fully-vectorized way (two matrix products, as in
    # ormYdeFittedMean()), not via rms's O(n)-memory streaming-over-k
    # loop -- that trick saves memory for a SINGLE evaluation, but here
    # the outer draw loop is unavoidable regardless, so a vectorized
    # n x k build per draw (freed after use) beats nesting a second,
    # k-length R-level loop inside an already-D-iteration one.
    k <- object$non.slopes
    ylevs <- ylevels
    alphaname <- if (k == 1) "Intercept" else paste0("y>=", ylevs[-1])
    # Claude Sonnet 5 2026-11-21
    # borm()'s own ytransmat (built above as ytrans(ylevels), the FULL
    # level set) has one row per Y level INCLUDING the first, unlike
    # rms::orm.fit()'s convention (one row per CUTPOINT, from
    # ylevels[-1]) -- confirmed from the existing 'lp'/'x' code just
    # above, whose rowidx <- match(ycut, ylevels) has no "-1" shift.
    # alpha[j] corresponds to ylevels[j+1] (confirmed from kint<-...;
    # ycut <- ylevels[kint+1] just above), so dropping ytransmat's own
    # first row aligns row j of the result with alpha[j], exactly as
    # rms::ormYdeFittedMean()'s ytransmat already is.
    ytransmat_cut <- ytransmat[-1, , drop = FALSE]  # k x myde
    draws  <- object$draws
    ndraws <- nrow(draws)
    betacols <- setdiff(colnames(draws), c(alphaname, taunam))
    Xbeta <- X %*% t(draws[, betacols, drop = FALSE]) # n x ndraws
    cumprob <- eval(rms::probabilityFamilies[[object$family]][1])

    if (type == "mean") {
      values <- if (codes) 1 : length(ylevs) else as.numeric(ylevs)
      if (any(is.na(values)))
        stop('values of response levels must be numeric for type="mean" and codes=FALSE')
      dvalues <- diff(values)
      R <- matrix(NA_real_, n, ndraws)
      for (d in 1 : ndraws) {
        Tau_d   <- matrix(draws[d, taunam], nrow = qyde, ncol = myde)
        ZTau_d  <- (Z %*% Tau_d) %*% t(ytransmat_cut)              # n x k
        lp_full <- sweep(ZTau_d, 2, draws[d, alphaname], "+") + Xbeta[, d]
        R[, d]  <- values[1] + as.vector(cumprob(lp_full) %*% dvalues)
      }
      if (posterior.summary == "all") return(t(R))
      postsum <- if (posterior.summary == "mean") rowMeans(R) else apply(R, 1, median)
      if (!cint) return(postsum)
      hpd <- apply(R, 1, HPDint, prob = cint)
      return(list(mean = postsum, lower = hpd[1, ], upper = hpd[2, ]))
    }

    ## type %in% c('fitted', 'fitted.ind'): only posterior.summary='mean'
    ## with no cint is supported so far -- 'median'/'all'/HPD would need
    ## every draw's full n x k table stored at once (ndraws*n*k), not
    ## just a running sum, a materially larger memory commitment not
    ## attempted in this pass. posterior.summary!='mean' is a deliberate
    ## ask and stops; cint, whose default (0.95, from predict.borm()'s
    ## own formal) is NOT a deliberate ask when the caller never
    ## mentioned it, only warns and is ignored -- stopping on cint's
    ## mere default would break plain predict(fit, nd, type='fitted').
    if (posterior.summary != "mean")
      stop("type='", type, "' for a borm() fit using yde/ytrans currently only ",
           "supports posterior.summary='mean' (median/all would need every ",
           "draw's full table stored at once) -- use type='lp' with ycut= ",
           "for a single-cutpoint posterior summary instead")
    if (cint)
      warning("cint is not yet supported for type='", type, "' on a borm() ",
              "fit using yde/ytrans (would need every draw's full table ",
              "stored at once, i.e. HPD per cell) -- returning the ",
              "posterior mean with no interval")
    Psum <- matrix(0, n, k)
    for (d in 1 : ndraws) {
      Tau_d   <- matrix(draws[d, taunam], nrow = qyde, ncol = myde)
      ZTau_d  <- (Z %*% Tau_d) %*% t(ytransmat_cut)
      lp_full <- sweep(ZTau_d, 2, draws[d, alphaname], "+") + Xbeta[, d]
      Psum    <- Psum + cumprob(lp_full)
    }
    P <- Psum / ndraws
    if (type == "fitted") {
      dimnames(P) <- list(rownames(X), alphaname)
      return(drop(P))
    }
    Peq <- cbind(1, P) - cbind(P, 0)
    ynam <- paste(object$yname, "=", ylevs, sep = "")
    dimnames(Peq) <- list(rownames(X), ynam)
    return(drop(Peq))
  }

  if (type == "x") {
    Zexp <- matrix(0., n, qyde * myde)
    for (i in 1:n) Zexp[i, ] <- expandZ(Z[i, ], ytransmat[rowidx[i], ])
    colnames(Zexp) <- taunam
    return(cbind(X, Zexp))
  }

  ## type == 'lp' (fun= already guarded against above). Uses the full
  ## posterior draws (object$draws already carries back-transformed
  ## alpha/beta/tau -- see borm()'s own draws <- cbind(alphas, betas,
  ## taus)), mirroring predict.blrm()'s own more general per-draw loop
  ## (its xb/ztau/r loop), generalized from a single tau column to
  ## tau's q*m flattened columns via expandZ().
  alphaname <- if (object$non.slopes == 1) "Intercept" else paste0("y>=", ylevels[-1])
  draws <- object$draws
  ndraws <- nrow(draws)
  Zexp <- matrix(0., n, qyde * myde)
  for (i in 1:n) Zexp[i, ] <- expandZ(Z[i, ], ytransmat[rowidx[i], ])
  # predictrms()'s own type='x' output uses model.matrix()'s raw naming
  # (e.g. "x2b"), not atr$colnames's rms-style naming (e.g. "x2=b"), so
  # betas must be matched to X positionally (what colnames remain in
  # draws after removing the alpha and tau columns, in order) rather
  # than by name -- the same convention blrm.r's own predict.blrm()
  # uses via matxv() (positional, not name-based).
  betacols <- setdiff(colnames(draws), c(alphaname, taunam))
  xb <- X %*% t(draws[, betacols, drop = FALSE]) # n x ndraws
  zt <- Zexp %*% t(draws[, taunam, drop = FALSE])   # n x ndraws
  r <- sweep(xb + zt, 2, draws[, alphaname[kint]], "+") # n x ndraws

  if (posterior.summary == "all") {
    return(t(r))
  }
  postsum <- if (posterior.summary == "mean") rowMeans(r) else apply(r, 1, median)
  if (!cint) {
    return(postsum)
  }
  hpd <- apply(r, 1, HPDint, prob = cint)
  list(linear.predictors = postsum, lower = hpd[1, ], upper = hpd[2, ])
}

# Claude Sonnet 5 2026-11-22
# Moved here (was between bormStats()'s own @seealso and @examples
# roxygen tags) -- found while debugging an unrelated issue that two
# function definitions sitting inside bormStats()'s roxygen comment
# block (the regular, non-##' comment above was not itself the
# problem) split it into two pieces: roxygen2 attaches a block to the
# next R expression it finds, so the title/description/@param/@return
# ended up attached to bormFmtPM() instead of bormStats(), and
# bormStats.Rd was generated with no title at all ("Skipping; no name
# and/or title"), confirmed via roxygen2::roxygenize() output. No
# content changed, only the location.
# Shared by both borm.r and blrm.r's own print.*/*Stats functions (one
# namespace across R/, per CLAUDE.md) for the two recurring
# uncertainty-interval string shapes: "point +/- se" and "point
# [lower, upper]". For non-plain (latex/html/typst) output, the point
# estimate stays at normal size; everything after it -- the +/- symbol
# and se, or the brackets/comma-space and lower/upper, numbers
# included -- is wrapped in markupSpecs[[lang]]$smaller2 (an earlier
# version wrapped the whole string including the point estimate,
# which Frank asked to revert; a version before that wrapped only the
# decorative symbols/brackets, leaving ALL numbers at normal size,
# which read as inconsistent). markupSpecs[[lang]]$smaller2 already
# exists for all three non-plain languages (Hmisc's html.s), so this
# reuses it rather than inventing new markup. $smaller2(...) is
# vectorized via paste0() the same way $smaller(...) was, so a single
# call on the already-built vector of
# strings wraps each element individually.
bormFmtPM <- function(point, se, dec = 2) {
  lang <- prType()
  pm <- if (lang == 'plain') '+/-' else markupSpecs[[lang]][['plminus']]
  tailstr <- paste0(pm, round(se, dec))
  if (lang != 'plain') tailstr <- markupSpecs[[lang]][['smaller2']](tailstr)
  paste0(round(point, dec), tailstr)
}

bormFmtCI <- function(point, lower, upper, dec = 3) {
  lang <- prType()
  tailstr <- paste0(' [', round(lower, dec), ', ', round(upper, dec), ']')
  if (lang != 'plain') tailstr <- markupSpecs[[lang]][['smaller2']](tailstr)
  paste0(round(point, dec), tailstr)
}

##' Compute Indexes of Predictive Accuracy and Their Uncertainties for `borm()`
##'
##' For an ordinal or binary regression fit from [borm()], computes several indexes of predictive accuracy along with highest posterior density intervals for them, closely paralleling [blrmStats()] (see its documentation for `g`, `gp`, `B`, `EV`, `v`, `vp`), but generalized two ways: the predicted-probability-scale quantities (`gp`, `B`, `EV`, `vp`) use the actual link family's cumulative distribution function (via `rms::probabilityFamilies`) rather than assuming logistic, and `Dxy`/`C` and the additional `rho` (Spearman's rank correlation between the linear predictor and Y) follow [rms::orm()]'s own censoring rules exactly: `rho` is `NA` whenever any censoring is present; `Dxy`/`C` are `NA` only when left- or interval-censoring is present (right-censoring alone is handled via [rms::Ocens2Surv()] and [survival::concordancefit()], the same mechanism [rms::orm.fit()] uses). For a fit that used `yde`/`ytrans`, `x=TRUE` must also have been given to [borm()] (it stores `z`, the `yde` design matrix, alongside `x`), and the linear predictor used throughout is evaluated at the fit's reference intercept's Y level, exactly mirroring [blrmStats()]'s own handling of a `cppo`-weighted `tau` -- there is no single well-defined "the" linear predictor for a Y-dependent-effect model otherwise.
##' @param fit an object produced by [borm()]
##' @param ns number of posterior draws to use in the calculations (default is 400)
##' @param prob HPD interval probability (default is 0.95)
##' @param pl set to `TRUE` to plot the posterior densities using base graphics
##' @param dist if `pl` is `TRUE` specifies whether to plot the density estimate (the default) or a histogram
##' @return list of class `bormStats` whose most important element is `Stats`
##' @seealso [blrmStats()], [Hmisc::rcorr.cens()]
##' @examples
##' \dontrun{
##'   f <- borm(...)
##'   bormStats(f, pl=TRUE)   # print and plot
##' }
##' @author Frank Harrell
##' @export
bormStats <- function(fit, ns = 400, prob = 0.95, pl = FALSE,
                      dist = c("density", "hist")) {
  dist <- match.arg(dist)

  f <- fit[c(
    "x", "y", "z", "non.slopes", "interceptRef", "draws", "yunique", "Ncens1", "family",
    "qyde", "ytrans"
  )]
  X <- f$x
  Y <- f$y
  if (length(X) == 0 | length(Y) == 0) {
    stop("must have specified x=TRUE, y=TRUE to borm")
  }
  anycens <- sum(f$Ncens1) > 0
  leftint <- isTRUE(unname(f$Ncens1["left"] > 0) || unname(f$Ncens1["interval"] > 0))
  if (is.matrix(Y)) {
    Sy <- if (anycens) Ocens2Surv(Y) else survival::Surv(Y[, 1])
    yv <- Y[, 1]
  } else {
    Sy <- survival::Surv(Y)
    yv <- Y
  }
  yv <- as.integer(yv) - 1
  nrp <- f$non.slopes
  kmid <- f$interceptRef
  draws <- f$draws
  ylev <- f$yunique

  # Claude Sonnet 5 2026-11-22          entire block
  # yde: draws has [alphas, betas, taus] when qyde>0 (see borm()'s own
  # draws <- cbind(alphas, betas, taus)); the old
  # s <- draws[, c(kmid, (nrp+1):ncol(draws)), drop=FALSE] silently
  # included the tau columns too, giving cbind(1,X) (p+1 columns)
  # non-conformable against beta (p+1+qyde*myde columns) -- confirmed
  # directly: print() on a yde-fitted borm() crashed with "non-
  # conformable arguments" (in cbind(1,X) %*% t(beta)) before this
  # fix. Mirrors blrmStats()'s own tauFetch()-based handling: the
  # Y-dependent contribution is evaluated AT THE REFERENCE INTERCEPT's
  # Y level (kmid), exactly as blrmStats() does for its own
  # cppo-weighted tau (and as predict.blrm()/Mean.blrm() default ycut
  # to that same reference level when not given) -- there is no
  # single well-defined "the" linear predictor for a Y-dependent-
  # effect model otherwise.
  qyde <- f$qyde
  if (!length(qyde)) qyde <- 0L
  Z <- f$z
  etau <- NULL
  if (qyde > 0) {
    cn <- colnames(draws)
    # Claude Sonnet 5 2026-11-22
    # Same fix as bormPredictYde()'s own: f$ytrans is already a real
    # function (not text) on a file= cache-hit object -- see that
    # function's own comment for the confirmed failure this avoids.
    ytransfn <- if (is.function(f$ytrans)) f$ytrans
                else eval(parse(text = f$ytrans))
    ytransmat <- ytransfn(ylev)
    # Claude Sonnet 5 2026-11-16          2 lines
    # ytrans is stored raw -- wrap a plain-vector result into a matrix,
    # exactly as borm()'s own construction of ytransmat already does.
    if (!is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)
    # Claude Sonnet 5 2026-11-23: myde is not stored -- recompute it
    myde <- ncol(ytransmat)
    taunam <- bormYdeNames(colnames(Z), ytransmat, qyde, myde)
    betanam <- setdiff(cn[-(1:nrp)], taunam)
    s <- draws[, c(cn[kmid], betanam), drop = FALSE]
    taudraws <- draws[, taunam, drop = FALSE]
    ytr <- ytransmat[kmid + 1, ]
    etau <- matrix(0., nrow(taudraws), qyde)
    for (i in 1:nrow(taudraws)) etau[i, ] <- matrix(taudraws[i, ], qyde, myde) %*% ytr
  } else {
    s <- draws[, c(kmid, (nrp + 1) : ncol(draws)), drop = FALSE]
  }
  ndraws <- nrow(s)
  ns <- min(ndraws, ns)
  if (ns < ndraws) {
    j <- sample(1:ndraws, ns, replace = FALSE)
    s <- s[j, , drop = FALSE]
    if (qyde > 0) etau <- etau[j, , drop = FALSE]
  }
  cumprob <- eval(rms::probabilityFamilies[[f$family]][1])

  # Claude Sonnet 5 2026-11-16          2 lines
  # pdm: the mean absolute difference between P(Y>=median) and 0.5,
  # exactly rms::orm.fit()'s own pdm <- mean(abs(cump(lp) - 0.5)) --
  # cumprob(lp) below is already this fit's P(Y>=median) per draw.
  stats <- matrix(NA, nrow = ns, ncol = 10)
  colnames(stats) <- c("Dxy", "C", "rho", "g", "gp", "B", "EV", "v", "vp", "pdm")
  dxy <- function(x, Syv) {
    con <- survival::concordancefit(Syv, x)$count
    conc <- con["concordant"]
    disc <- con["discordant"]
    (conc - disc) / (conc + disc)
  }
  brier <- function(x, y) mean((x - y)^2)
  br2 <- function(p) var(p) / (var(p) + sum(p * (1 - p)) / length(p))

  nobs <- length(yv)
  is <- if ((nobs <= 10000) || (length(ylev) == 2)) {
    1:nobs
  } else {
    sample(1:nobs, 10000, replace = FALSE)
  }

  for (i in 1:ns) {
    beta <- s[i, , drop = FALSE]
    lp <- cbind(1, X) %*% t(beta)
    if (qyde > 0) lp <- lp + Z %*% etau[i, ]
    prb <- cumprob(lp)
    d <- if (anycens && leftint) NA_real_ else dxy(lp[is], Sy[is])
    C <- if (is.na(d)) NA_real_ else (d + 1.) / 2.
    rho <- if (anycens) {
      NA_real_
    } else if (diff(range(lp)) == 0e0) {
      0e0
    } else {
      cor(rank(lp[is]), rank(yv[is]))
    }
    st <- c(
      d, C, rho, GiniMd(lp), GiniMd(prb),
      brier(prb, yv > kmid - 1),
      br2(prb), var(lp), var(prb), mean(abs(prb - 0.5))
    )
    stats[i, ] <- st
  }
  sbar <- colMeans(stats, na.rm = TRUE)
  se <- apply(stats, 2, sd, na.rm = TRUE)
  hpd <- apply(stats, 2, function(z) if (all(is.na(z))) c(NA_real_, NA_real_) else HPDint(z[!is.na(z)], prob = prob))
  sym <- apply(stats, 2, function(z) if (all(is.na(z))) NA_real_ else distSym(z[!is.na(z)]))

  text <- bormFmtCI(sbar, hpd[1, ], hpd[2, ], dec = 3)
  Stats <- rbind(Mean = sbar, SE = se, hpd = hpd, Symmetry = sym)
  statnames <- colnames(Stats)
  names(text) <- statnames

  if (pl) {
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar))
    par(mfrow = c(3, 3), mar = c(3, 2, 0.5, 0.5), mgp = c(1.75, .55, 0))
    for (w in setdiff(statnames, "C")) {
      zz <- stats[, w]
      zz <- zz[!is.na(zz)]
      if (!length(zz)) next
      p <- switch(dist,
        density = {
          den <- density(zz)
          plot(den, xlab = w, ylab = "", type = "l", main = "")
        },
        hist = {
          den <- hist(zz,
            probability = TRUE,
            nclass = 100, xlab = w, ylab = "", main = ""
          )
          den$x <- den$breaks
          den$y <- den$density
        }
      )
      ref <- c(sbar[w], hpd[1, w], hpd[2, w])
      abline(v = ref, col = gray(0.85))
      text(min(den$x), max(den$y), paste0("Symmetry:", round(sym[w], 2)),
        adj = c(0, 1), cex = 0.7
      )
    }
  }
  structure(list(
    stats = Stats, text = text, ndraws = ndraws, ns = ns,
    non.slopes = nrp, intercept = kmid
  ), class = "bormStats")
}

##' Print Details for `bormStats` Predictive Accuracy Measures
##'
##' Prints results of `bormStats` with brief explanations
##' @param x an object produced by `bormStats`
##' @param dec number of digits to round indexes
##' @param ... ignored
##' @author Frank Harrell
##' @export
print.bormStats <- function(x, dec = 3, ...) {
  ns <- x$ns
  ndraws <- x$ndraws
  cat("Indexes computed on a sample of", ns, "of the", ndraws, "posterior draws\n\n")
  print(round(x$stats, dec))
  cat("\nDxy: Somers' Dxy rank correlation; C: concordance probability; rho: Spearman correlation\n")
  cat("g, gp: Gini's mean difference (linear predictor, probability scales)\n")
  cat("B: Brier score; EV: explained variation; v, vp: variance (linear predictor, probability scales)\n")
  invisible()
}

##' Print a `borm()` Fit
##'
##' Prints a `borm()` fit object, with a family-specific title (paralleling [rms::print.orm()]'s own family-dependent title) and a Dirichlet-prior subtitle, otherwise following [print.blrm()]'s layout closely, including [bormStats()]'s discrimination/calibration indexes (now including `rho`, Spearman's rank correlation, alongside `Dxy`/`C`).
##' @param x a [borm()] fit object
##' @param dec number of digits to the right of the decimal place
##' @param coefs set to `FALSE` to suppress printing the coefficient table
##' @param intercepts set to `FALSE` to suppress printing intercepts in the coefficient table; default is to print them when there are fewer than 10
##' @param prob the probability for highest posterior density intervals
##' @param ns number of posterior draws passed to [bormStats()] for the discrimination/calibration statistics
##' @param title an optional title overriding the default family-specific one
##' @param ... ignored
##' @return nothing; prints
##' @author Frank Harrell
##' @method print borm
##' @export
print.borm <- function(x, dec = 4, coefs = TRUE, intercepts = x$non.slopes < 10,
                       prob = 0.95, ns = 400, title = NULL, ...) {
  latex <- prType() == "latex"

  if (!length(title)) {
    # Claude Sonnet 5 2026-11-19          entire block
    # Family-dependent title, matching rms::print.orm()'s own switch
    # exactly (prefixed "Bayesian", as blrm's titles are).
    famtitle <- switch(x$family,
      logistic = "Logistic (Proportional Odds)",
      probit   = "Probit",
      cauchit  = "Cauchy",
      loglog   = "-log-log",
      cloglog  = "Complementary log-log"
    )
    title <- paste("Bayesian", famtitle, "Ordinal Regression Model")
  }
  conc <- x[["conc"]]
  subtitle <- paste(
    "Dirichlet Priors With Concentration Parameter",
    round(conc, 3), "for Intercepts"
  )

  z <- list()

  if (length(x$freq) > 3 && length(x$freq) < 50)
    z <- c(z, list(rms::prModItem(
      "print", list(x$freq),
      title = "Frequencies of Responses"
    )))
  if (length(x$na.action))
    z <- c(z, list(rms::prModItem(
      paste("naprint", class(x$na.action), sep = "."),
      list(x$na.action)
    )))
  qro <- function(x) {
    hpd <- HPDint(x, prob)
    bormFmtCI(median(x), hpd[1], hpd[2], dec = 4)
  }
  ci <- x$clusterInfo
  sigmasum <- sigma1sum <- sigma2sum <- NULL
  if (length(ci)) {
    if ("sigma1" %in% colnames(x$omega)) {
      sigma1sum <- qro(x$omega[, "sigma1"])
      sigma2sum <- qro(x$omega[, "sigma2"])
    } else {
      sigmasum <- qro(x$omega[, "sigmag"])
    }
  }

  loo <- x$loo
  elpd_loo <- p_loo <- looic <- NULL
  if (length(loo)) {
    lo <- loo$estimates
    lo <- bormFmtPM(lo[, "Estimate"], lo[, "SE"], dec = 2)
    elpd_loo <- lo[1]
    p_loo <- lo[2]
    looic <- lo[3]
  }
  stime <- if (length(x$sampling_time)) paste0(round(x$sampling_time, 1), "s")

  misc <- rms::reListclean(
    Obs = x$N,
    "Distinct Y" = length(x$freq),
    "Median Y" = x$mediany,
    p = x$p,
    Draws = nrow(x$draws),
    Chains = x$chains,
    Time = stime,
    "Cluster on" = ci$name,
    Clusters = ci$n,
    "sigma gamma" = sigmasum,
    sigma1 = sigma1sum,
    sigma2 = sigma2sum
  )

  if (length(x$freq) < 4) {
    names(x$freq) <- paste(if (latex) "~~" else " ",
      names(x$freq),
      sep = ""
    )
    misc <- c(misc[1], x$freq, misc[-1])
  }
  a <- bormStats(x, ns = ns)$text
  mixed <- rms::reListclean(
    "LOO log L" = elpd_loo,
    "LOO IC" = looic,
    "Effective p" = p_loo,
    B = a["B"]
  )

  disc <- rms::reListclean(
    "|P(Y>=median)-0.5|" = a["pdm"]
  )

  discr <- rms::reListclean(
    rho = a["rho"],
    C = a["C"],
    Dxy = a["Dxy"]
  )

  headings <- c(
    "", "Mixed Calibration/\nDiscrimination Indexes",
    "Discrimination\nIndexes",
    "Rank Discrim.\nIndexes"
  )

  data <- list(misc, c(mixed, NA), c(disc, NA), c(discr, NA))
  # Claude Sonnet 5 2026-11-22
  # Plain-text-only per Frank's request: widen the first column (misc
  # -- Obs/Distinct Y/.../Draws/Chains/Time) by 3 characters and
  # narrow "Discrimination Indexes" (disc) by 5; the other two columns
  # are unaffected. See prStats()'s own widthadj= argument.
  widthadj <- c(3, 0, -5, 0)
  z <- c(z, list(rms::prModItem(
    "stats", list(headings = headings, data = data, widthadj = widthadj)
  )))

  if (coefs)
    z <- c(z, list(rms::prModItem(
      "coefmatrix",
      list(bayes = print.rmsb(x,
        prob = prob, intercepts = intercepts,
        pr = FALSE
      ))
    )))

  if (length(x$pcontrast))
    z <- c(z, list(rms::prModItem(
      "print", list(deparse(x$pcontrast)),
      title = "Contrasts Given Priors"
    )))

  rms::prModFit(x,
    title = title, z, digits = dec, coefs = coefs,
    subtitle = subtitle, ...
  )
}

# Claude Sonnet 5 2026-11-19          entire block
# predict.blrm/Mean.blrm/Quantile.blrm/ExProb.blrm are already
# link-and-model-structure-agnostic: they dispatch through
# object$link + rms::probabilityFamilies (not hardcoded to logistic),
# and their only ppo/cppo-specific logic is guarded behind
# `object$pppo > 0`/`length(object$cppo)` -- confirmed directly by
# inspection (no hardcoded "blrm" class check inside any of the four
# function bodies that would make this unsafe). borm()'s own fit
# object does not carry pppo/cppo/ylevels/link itself (its own fields
# are qyde/yunique/family -- see bormBlrmDelegate(), updated 2026-11-23
# to also bridge ylevels/link, not just pppo/cppo); the four delegation
# call sites below attach all of these to a local copy of the object
# just for the delegated call. Rather than duplicate ~400 lines of
# identical logic, borm's own S3 methods simply delegate to them.
##' Make Predictions from a `borm()` Fit
##'
##' Predict method for [borm()] objects. When the fit did not use `yde`/`ytrans`, delegates entirely to [predict.blrm()], which is already link-and-model-structure-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`, with its only `ppo`/`cppo`-specific logic guarded behind fields `borm()` always sets to 0/`NULL`). When the fit did use `yde`/`ytrans`, has its own `yde`-aware implementation for `type='lp'` and `type='x'` (the two most commonly needed, and the building blocks `type='fitted'` etc. use internally); other types, and `fun=`, are not yet implemented for a `yde`-fitted model and raise an informative error rather than silently ignoring the Y-dependent effect, since [predict.blrm()]'s own (hardcoded single-column `cppo`) logic cannot be reused unchanged for `borm()`'s matrix-valued `tau`.
##' @param object a [borm()] fit object
##' @param ... passed to [predict.blrm()], or (for a `yde`-fitted model) to [rms::predictrms()]
##' @param kint see [predict.blrm()]
##' @param ycut for a `yde`-fitted model, the Y value(s) at which to evaluate the Y-dependent effect; see [blrm()]'s own `ycut` documentation under `predict.blrm`. Defaults to the fit's reference intercept's Y level, exactly as [predict.blrm()] does.
##' @param fun see [predict.blrm()]; not yet implemented for a `yde`-fitted model (errors if given)
##' @param funint see [predict.blrm()]
##' @param type see [predict.blrm()]; for a `yde`-fitted model only `'lp'` and `'x'` are currently implemented (errors otherwise)
##' @param se.fit see [predict.blrm()]
##' @param codes see [predict.blrm()]
##' @param posterior.summary see [predict.blrm()]
##' @param cint see [predict.blrm()]
##' @return see [predict.blrm()]
##' @seealso [predict.blrm()]
##' @author Frank Harrell
##' @method predict borm
##' @export
predict.borm <- function(object, ...,
                         kint = NULL, ycut = NULL,
                         fun = NULL, funint = TRUE,
                         type = c(
                           "lp", "fitted", "fitted.ind", "mean", "x", "data.frame",
                           "terms", "cterms", "ccterms", "adjto", "adjto.data.frame",
                           "model.frame"
                         ),
                         se.fit = FALSE, codes = FALSE,
                         posterior.summary = c("mean", "median", "all"), cint = 0.95) {
  qyde <- object$qyde
  if (!length(qyde)) qyde <- 0L
  if (qyde == 0) {
    return(predict.blrm(bormBlrmDelegate(object), ...,
      kint = kint, ycut = ycut, fun = fun, funint = funint,
      type = match.arg(type), se.fit = se.fit, codes = codes,
      posterior.summary = match.arg(posterior.summary), cint = cint
    ))
  }
  bormPredictYde(object, ...,
    kint = kint, ycut = ycut, fun = fun, funint = funint,
    type = match.arg(type), se.fit = se.fit, codes = codes,
    posterior.summary = match.arg(posterior.summary), cint = cint
  )
}

##' `Mean()` Closure for a Y-Dependent-Effect `borm()` Fit
##'
##' Internal helper for [Mean.borm()] when `qyde > 0`, the `rmsb` parallel
##' of `rms`'s own `ormYdeMean()` (ported as an algorithm, not a shared
##' call, per the explicit cross-package modularization decision -- see
##' `orm-yde-implementation.md` Sec. 14.5). [Mean.blrm()]'s own closure
##' takes a bare, already-posterior-summarized `lp`/`lptau` -- structurally
##' insufficient for a `yde` fit for the same reason as the `orm()` side:
##' the Y-dependent effect varies by cutpoint, so no single `lp` value can
##' represent it. Requires `newdata` and delegates to `predict(object,
##' newdata, type='mean', ...)`, which does the real per-posterior-draw
##' work (see `bormPredictYde()`).
##' @param object a `yde`-fitted [borm()] object
##' @param codes see [Mean.blrm()]'s own `codes` argument
##' @author Frank Harrell
##' @noRd
bormYdeMean <- function(object, codes = FALSE) {
  function(newdata = NULL, posterior.summary = c('mean', 'median', 'all'),
           cint = 0, ...) {
    if (! length(newdata))
      stop("Mean() for a borm() fit using yde/ytrans must be called with ",
           "newdata (the original covariate data frame) -- a bare linear ",
           "predictor cannot represent a Y-dependent effect, which varies ",
           "by level of Y. Call this function as m(newdata=...), or use ",
           "predict(object, newdata, type='mean') directly.")
    posterior.summary <- match.arg(posterior.summary)
    predict(object, newdata, type = 'mean', codes = codes,
            posterior.summary = posterior.summary, cint = cint)
  }
}

##' `Quantile()` Closure for a Y-Dependent-Effect `borm()` Fit
##'
##' Internal helper for [Quantile.borm()] when `qyde > 0`, paralleling
##' `bormYdeMean()` and `rms::ormYdeQuantile()`. Requires `newdata` and
##' inverts `P <- predict(object, newdata, type='fitted', ...)` (only
##' `posterior.summary='mean'` is available for `type='fitted'` on a
##' `yde` fit -- see `bormPredictYde()`) via the same `approx()`-based
##' logic `rms::Quantile.orm()`'s own (and `ormYdeQuantile()`'s)
##' `method='discrete'`/`'interpolated'` use, ported unchanged apart from
##' sourcing `P` from `predict(type='fitted')` instead of `cumprob(lb)`
##' (`P` being a posterior mean across draws rather than a single MLE
##' evaluation doesn't affect either method's math: a posterior mean of
##' per-draw monotone-in-Y probability vectors is itself monotone, which
##' is all `'interpolated'`'s grid construction relies on -- confirmed
##' numerically, not just argued, in the verification for this function).
##' @param object a `yde`-fitted [borm()] object
##' @param codes see [Quantile.blrm()]'s own `codes` argument
##' @author Frank Harrell
##' @noRd
bormYdeQuantile <- function(object, codes = FALSE) {
  function(q = .5, newdata = NULL, posterior.summary = 'mean',
           method = c('interpolated', 'discrete'), ...) {
    if (! length(newdata))
      stop("Quantile() for a borm() fit using yde/ytrans must be called with ",
           "newdata (the original covariate data frame) -- a bare linear ",
           "predictor cannot represent a Y-dependent effect, which varies ",
           "by level of Y. Call this function as f(newdata=...), or use ",
           "predict(object, newdata, type='fitted') directly.")
    if (! identical(posterior.summary, 'mean'))
      stop("Quantile() for a borm() fit using yde/ytrans only supports ",
           "posterior.summary='mean'")
    method <- match.arg(method)
    values <- if (codes) 1 : length(object$yunique) else as.numeric(object$yunique)
    if (any(is.na(values)))
      stop('values of response levels must be numeric for codes=FALSE')
    ns <- length(values) - 1L
    # Claude Sonnet 5 2026-11-22
    # cint=0 explicitly: predict.borm()'s own cint formal defaults to
    # 0.95, which would otherwise trigger bormPredictYde()'s "cint is
    # not yet supported for type='fitted'" warning on every call here,
    # even though nothing about this closure ever asked for an interval.
    P <- predict(object, newdata, type = 'fitted', codes = codes,
                 posterior.summary = 'mean', cint = 0)
    if (! is.matrix(P)) P <- matrix(P, nrow = 1)
    n <- nrow(P)

    if (method == 'interpolated') {
      pfam    <- rms::probabilityFamilies[[object$family]]
      cumprob <- eval(pfam[1])
      inverse <- eval(pfam[2])
      m.yvals <- matrix(NA, nrow = n, ncol = ns + 2)
      cp <- cbind(P, 0)
      for (j in 1 : n) {
        ws <- c(0, (cp[j, -ns-1] - cp[j, 1]) / (cp[j, ns] - cp[j, 1]), 1)
        m.yvals[j,] <- (1 - ws) * c(values[1], values) + ws * c(values, values[ns + 1])
      }
      z <- sapply(1 : n,
                 function(i) approx(c(1, cp[i,]), m.yvals[i,],
                                    xout = cumprob(inverse(1 - q)), rule = 2)$y)
    } else {
      m.cdf <- cbind(1 - P, 1)
      id <- apply(m.cdf, 1, FUN = function(x) {min(which(x >= q))[1]})
      z  <- values[id]
    }
    names(z) <- rownames(P)
    z
  }
}

##' `ExProb()` Closure for a Y-Dependent-Effect `borm()` Fit
##'
##' Internal helper for [ExProb.borm()] when `qyde > 0`, paralleling
##' `bormYdeMean()` and `rms::ormYdeExProb()`. Requires `newdata` and
##' sources the per-cutpoint exceedance probabilities from `predict(
##' object, newdata, type='fitted', posterior.summary='mean')` (the only
##' `posterior.summary` available for `type='fitted'` on a `yde` fit --
##' see `bormPredictYde()`), prepending the trivial `P(Y>=values[1])=1`
##' column exactly as `ormYdeExProb()` does. The rest -- the `y=NULL`
##' vs. `y=`-given branching, the step-function lookup, the `'ExProb'`-
##' class return shape -- is copied from [ExProb.blrm()]/`ormYdeExProb()`,
##' since none of it is `yde`-specific.
##' @param object a `yde`-fitted [borm()] object
##' @param codes see [ExProb.blrm()]'s own `codes` argument
##' @author Frank Harrell
##' @noRd
bormYdeExProb <- function(object, codes = FALSE) {
  # Claude Sonnet 5 2026-11-22          4 lines
  # conf.int= is an explicit, checked formal (not swallowed by ...)
  # specifically so a request for it stops with a clear message --
  # confirmed directly that ExProb.blrm()'s own (non-yde) closure has
  # no ... at all, so the same request there errors with R's generic
  # "unused argument"; silently ignoring it here (an earlier, less
  # careful version of this function did, via its own ...) would be
  # less informative than that, not just inconsistent with it.
  function(y = NULL, newdata = NULL, posterior.summary = 'mean', conf.int = 0, ...) {
    if (! length(newdata))
      stop("ExProb() for a borm() fit using yde/ytrans must be called with ",
           "newdata (the original covariate data frame) -- a bare linear ",
           "predictor cannot represent a Y-dependent effect, which varies ",
           "by level of Y. Call this function as f(newdata=...), or use ",
           "predict(object, newdata, type='fitted') directly.")
    if (! identical(posterior.summary, 'mean'))
      stop("ExProb() for a borm() fit using yde/ytrans only supports ",
           "posterior.summary='mean'")
    if (conf.int)
      stop("conf.int is not yet implemented for ExProb() on a borm() fit ",
           "using yde/ytrans")
    values <- if (codes) 1 : length(object$yunique) else as.numeric(object$yunique)
    if (any(is.na(values)))
      stop('values of response levels must be numeric for codes=FALSE')
    yname <- object$yname
    # Claude Sonnet 5 2026-11-22
    # cint=0 explicitly -- see bormYdeQuantile()'s own identical comment.
    P <- predict(object, newdata, type = 'fitted', codes = codes,
                 posterior.summary = 'mean', cint = 0)
    if (! is.matrix(P)) P <- matrix(P, nrow = 1)
    prob <- cbind(1, P)
    if (! length(y)) {
      colnames(prob) <- paste('Prob(Y>=', values, ')', sep = '')
      result <- structure(list(y = values, prob = prob, yname = yname), class = 'ExProb')
    } else {
      p <- t(apply(prob, 1,
                 function(probs) {
                   pa <- approx(values, probs, xout = y, f = 1, method = 'constant')$y
                   pa[y < min(values)] <- 1.
                   pa[y > max(values)] <- 0.
                   pa
                 } ) )
      if (length(y) > 1) {
        colnames(p) <- paste('Prob(Y>=', y, ')', sep = '')
        p <- list(y = y, prob = p)
      }
      result <- structure(drop(p), class = 'ExProb')
    }
    result
  }
}

##' Function Generator for Mean Y for `borm()`
##'
##' For an ordinary (non-`yde`) fit, delegates entirely to [Mean.blrm()],
##' which is already link-agnostic (dispatches via `object$link`/
##' `rms::probabilityFamilies`). See [Mean.blrm()] for details. For a
##' `yde` fit, returns `bormYdeMean()`'s closure instead, which requires
##' `newdata` rather than a bare linear predictor.
##' @param object a [borm()] fit object
##' @param ... passed to [Mean.blrm()] for a non-`yde` fit; `codes` is
##'   read from here for a `yde` fit
##' @return see [Mean.blrm()] (non-`yde`) or `bormYdeMean()` (`yde`)
##' @seealso [Mean.blrm()]
##' @author Frank Harrell
##' @method Mean borm
##' @export
Mean.borm <- function(object, ...) {
  if (length(object$qyde) && object$qyde > 0) {
    dots <- list(...)
    return(bormYdeMean(object, codes = isTRUE(dots$codes)))
  }
  Mean.blrm(bormBlrmDelegate(object), ...)
}

##' Function Generator for Quantiles of Y for `borm()`
##'
##' For an ordinary (non-`yde`) fit, delegates entirely to [Quantile.blrm()],
##' which is already link-agnostic. See [Quantile.blrm()] for details. For
##' a `yde` fit, returns `bormYdeQuantile()`'s closure instead, which
##' requires `newdata` rather than a bare linear predictor.
##' @param object a [borm()] fit object
##' @param ... passed to [Quantile.blrm()] for a non-`yde` fit; `codes` is
##'   read from here for a `yde` fit
##' @return see [Quantile.blrm()] (non-`yde`) or `bormYdeQuantile()` (`yde`)
##' @seealso [Quantile.blrm()]
##' @author Frank Harrell
##' @method Quantile borm
##' @export
Quantile.borm <- function(object, ...) {
  if (length(object$qyde) && object$qyde > 0) {
    dots <- list(...)
    return(bormYdeQuantile(object, codes = isTRUE(dots$codes)))
  }
  Quantile.blrm(bormBlrmDelegate(object), ...)
}

##' Exceedance Probability Function Generator for `borm()`
##'
##' For an ordinary (non-`yde`) fit, delegates entirely to [ExProb.blrm()],
##' which is already link-agnostic. See [ExProb.blrm()] for details. For
##' a `yde` fit, returns `bormYdeExProb()`'s closure instead, which
##' requires `newdata` rather than a bare linear predictor.
##' @param object a [borm()] fit object
##' @param ... passed to [ExProb.blrm()] for a non-`yde` fit; `codes` is
##'   read from here for a `yde` fit
##' @return see [ExProb.blrm()] (non-`yde`) or `bormYdeExProb()` (`yde`)
##' @seealso [ExProb.blrm()]
##' @author Frank Harrell
##' @method ExProb borm
##' @export
ExProb.borm <- function(object, ...) {
  if (length(object$qyde) && object$qyde > 0) {
    dots <- list(...)
    return(bormYdeExProb(object, codes = isTRUE(dots$codes)))
  }
  ExProb.blrm(bormBlrmDelegate(object), ...)
}
