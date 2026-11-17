##' Bayesian Ordinal Regression
##'
##' Uses `rstan` with pre-compiled Stan code, or `cmdstan`, to get posterior draws of parameters from a Bayesian ordinal (or binary) cumulative probability regression model, with a choice of 5 link families -- `borm()` is to [blrm()] as [rms::orm()] is to [rms::lrm()]: it generalizes the single (logistic) link that `blrm()` implements to all of the link families `orm()` supports, and is designed, like `orm()`, to scale to a very large number of distinct Y values.  Unlike `blrm()`, `borm()` does not implement the partial proportional odds model (`ppo`/`cppo`) itself; `blrm()` remains the function to use for that. Instead `borm()` implements a more general Y-dependent effect mechanism that plays `cppo`'s role, via two new arguments: `yde` (paralleling `ppo`) identifies which predictors get a Y-dependent effect, and `ytrans` (paralleling `cppo`, but returning a matrix of one or more columns instead of a single score) specifies the shape(s) of that dependency. A single-column `ytrans` with a single `yde` predictor reduces mathematically to `blrm()`'s own `cppo` mechanism.
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
##' @param ytrans a function of one argument, `y`, specifying the shape(s) of the Y-dependent effect(s) for the predictors in `yde`, paralleling [blrm()]'s own `cppo` argument but generalized to allow more than one shape at once: it is called on the distinct, ordered Y values seen in the data (`y`'s own values need not be the complete set, or in any particular order -- `ytrans` must work correctly on an arbitrary subset/order of Y values, since it is also called later, e.g. for contrasts, on individual Y values) and must return either a plain vector (one Y-dependent shape) or a matrix with one or more columns (several simultaneous Y-dependent shapes), one row (or value) per element of `y`. `ytrans` is used exactly as written for every purpose a user or `borm()`'s own methods will see (storage, `bormStats()`, `predict.borm()`), with no centering -- so `tau`, the coefficient `borm()` reports, is directly interpretable as "per unit of `ytrans(y)`, per unit of the `yde` predictor," independent of the sample means of `y`, the `yde` predictors, or any other covariate; an earlier version centered `ytrans` automatically, which was reversed after confirming directly that it silently distorted the *other*, ordinary (non-`yde`) coefficient(s) for any predictor appearing in both `yde` and the main formula -- the common case, since a Y-dependent effect is normally an extension of an ordinary effect, not a separate one -- while providing no measurable sampling benefit. Internally (invisible to the user, and not reflected in the stored/reported `tau`), `ytrans(y)` IS still divided by its own SD over the observed Y values before being given to Stan, purely for numerical safety -- confirmed directly that skipping this is not safe: with no scaling at all, a `ytrans` producing values in roughly the hundreds or more (`ytrans=function(y) 100*y` on data with `y` up to about 10, for example) crashed R outright (a Stan optimizer segfault), while values up to the tens did not. A single-column `ytrans` applied to a single `yde` predictor reduces mathematically to `blrm()`'s own `cppo` mechanism. Must be accompanied by `yde`.
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

  fit.object <- list(
    call = call, fitter = fitter, link = family, family = family,
    yname = yname, ylevels = ylev, freq = numy, mediany = mediany,
    pppo = 0L, cppo = NULL, # degrades blrm()-shared predict/Mean/Quantile/ExProb code correctly
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
    # Claude Sonnet 5 2026-11-21          4 lines
    # yde: qyde/myde (0 when yde absent) and the (possibly wrapped,
    # center/scale-baked-in) ytrans deparsed as text -- same convention
    # as blrm's own cppo=deparse(cppo), to avoid serializing a huge
    # captured environment; re-hydrated with eval(parse(text=...)) the
    # same way blrm.r re-hydrates a cached cppo (see borm()'s own file
    # caching block above).
    yde = yde, qyde = qyde, myde = myde,
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
##' Internal helper for [borm()]: labels for `tau`'s columns, using `ytransmat`'s own column names when the user's `ytrans` function supplied them, else generic `ytrans1, ytrans2, ...` labels.
##' @param ytransmat the `k x myde` `ytrans(ylev)` matrix
##' @param myde number of `ytrans` columns
##' @return a length-`myde` character vector
##' @author Frank Harrell
##' @noRd
bormYdeColLab <- function(ytransmat, myde) {
  cn <- colnames(ytransmat)
  # Claude Sonnet 5 2026-11-21          1 line
  # Reject names cbind() auto-infers from a bare-symbol argument (e.g.
  # ytrans <- function(y) cbind(y, y^2) gives colnames c("y", "") --
  # confirmed directly), not a deliberate user label; an empty or
  # duplicated name is diagnostic of this, a fully-named, unique set is
  # not and is trusted as deliberate.
  if (!length(cn) || any(cn == "") || anyDuplicated(cn)) cn <- paste0("ytrans", 1:myde)
  cn
}

##' Flattened Names for a borm() yde tau Matrix
##'
##' Internal helper for [borm()]: names for `tau`'s flattened (column-major, matching `as.vector()`/Stan's own convention) entries, as `"<Z column> x <ytrans column>"`.
##' @param Zcolnames column names of the `yde` design matrix `Z`
##' @param ytransmat the `k x myde` `ytrans(ylev)` matrix
##' @param qyde number of `yde` predictors
##' @param myde number of `ytrans` columns
##' @return a length-`qyde*myde` character vector
##' @author Frank Harrell
##' @noRd
bormYdeNames <- function(Zcolnames, ytransmat, qyde, myde) {
  paste(rep(Zcolnames, myde), "x", rep(bormYdeColLab(ytransmat, myde), each = qyde))
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

##' Y-Dependent-Effect-Aware Prediction for a `borm()` Fit Using `yde`/`ytrans`
##'
##' Internal helper for [predict.borm()]: implements `type='lp'` (the posterior mean/median point-estimate case, without `cint` or `fun=`) and `type='x'`, generalizing [predict.blrm()]'s own single-column `cppo` logic (`object$cppo`, a scalar `Z*cppo(ycut)` term) to `borm()`'s `q x m` `tau` matrix and `k x m` `ytransmat`: the Y-dependent design contribution for one observation at one Y cutoff is the `q*m`-length vector `as.vector(outer(Z[row,], ytransmat[cutoff row,]))` (in the same column-major order `tau`'s flattened draws/back-transform use -- see `bormYdeBT()`'s own documentation), dotted against the flattened `tau` coefficients. `newdata` is resolved via [rms::gendata()] exactly as [rms::Xcontrast()] does, since [rms::predictrms()]'s own `second=TRUE` mechanism cannot be reused here (it is keyed off `object$pppo`, which `borm()` always sets to 0 so that `blrm()`'s shared `predict`/`Mean`/`Quantile`/`ExProb` code degrades correctly for a non-`yde` `borm()` fit).
##' @author Frank Harrell
##' @noRd
bormPredictYde <- function(object, ...,
                            kint, ycut, fun, funint, type,
                            se.fit, codes, posterior.summary, cint) {
  if (type %nin% c("lp", "x")) {
    stop(
      "type='", type, "' is not yet implemented for a borm() fit using yde/ytrans; ",
      "only type='lp' and type='x' are currently supported for such fits"
    )
  }
  if (length(fun)) stop("fun= is not yet implemented for a borm() fit using yde/ytrans")

  dots <- list(...)
  nd <- bormGetNewdata(object, dots)

  X <- rms::predictrms(object, newdata = nd, type = "x")
  n <- nrow(X)

  zatr <- object$zDesign
  # Claude Sonnet 5 2026-11-21          entire block
  # Building Z at predict time needs two fixes to rms::Design(), both
  # confirmed directly with standalone reproductions, neither of which
  # is about yde/ytrans logic itself: (1) a data frame containing
  # columns beyond those yde's own formula references -- e.g. the main
  # formula's own x alongside yde's x2 -- makes rms::Design() fail
  # ("program logic error tl") once more than one newdata row is
  # requested, apparently via an interaction with the active global
  # datadist registering both variables; fixed by subsetting to only
  # yde's own variables (all.vars(object$yde)) first. (2) separately,
  # rms::Design() cannot build contrasts for a categorical predictor
  # from newdata containing only one of its levels (confirmed: "x2 has
  # <2 category levels" on a plain 1-row data frame), unlike
  # predictrms()'s own machinery for the MAIN formula, which reuses
  # object$Design's stored level information instead of re-deriving it
  # from newdata; fixed by explicitly releveling factor/character
  # columns to the full level set seen when yde was fit (zatr$values).
  nd2 <- bormGetNewdata(object, dots)
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

  qyde <- object$qyde
  myde <- object$myde
  # ytrans is stored deparsed (object$ytrans = deparse(ytrans) in
  # borm() itself), the same convention blrm.r uses for cppo, to avoid
  # serializing a captured environment; re-hydrate it here exactly as
  # the file-cache-hit block in borm() does.
  ytrans <- eval(parse(text = object$ytrans))
  ylevels <- object$ylevels
  ytransmat <- ytrans(ylevels)
  # Claude Sonnet 5 2026-11-16          2 lines
  # ytrans is stored raw (no more auto-centering/scaling wrapper that
  # used to guarantee a matrix return) -- wrap a plain-vector result,
  # exactly as borm()'s own construction of ytransmat already does.
  if (!is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)

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
    "x", "y", "z", "non.slopes", "interceptRef", "draws", "ylevels", "Ncens1", "link",
    "qyde", "myde", "ytrans"
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
  ylev <- f$ylevels

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
  myde <- f$myde
  Z <- f$z
  etau <- NULL
  if (qyde > 0) {
    cn <- colnames(draws)
    ytransmat <- eval(parse(text = f$ytrans))(ylev)
    # Claude Sonnet 5 2026-11-16          2 lines
    # ytrans is stored raw -- wrap a plain-vector result into a matrix,
    # exactly as borm()'s own construction of ytransmat already does.
    if (!is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)
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
  cumprob <- eval(rms::probabilityFamilies[[f$link]][1])

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

  text <- paste0(
    round(sbar, 3), " [", round(hpd[1, ], 3), ", ",
    round(hpd[2, ], 3), "]"
  )
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
    r <- round(c(median(x), HPDint(x, prob)), 4)
    paste0(r[1], " [", r[2], ", ", r[3], "]")
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
    pm <- if (prType() == "plain") {
      "+/-"
    } else {
      markupSpecs[[prType()]][["plminus"]]
    }
    lo <- paste0(round(lo[, "Estimate"], 2), pm, round(lo[, "SE"], 2))
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
  z <- c(z, list(rms::prModItem("stats", list(headings = headings, data = data))))

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
# `object$pppo > 0`/`length(object$cppo)`, which borm()'s fit object
# always sets to 0/NULL -- confirmed directly by inspection (no
## hardcoded "blrm" class check inside any of the four function bodies
# that would make this unsafe). Rather than duplicate ~400 lines of
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
                           "lp", "fitted", "fitted.ind", "x", "data.frame",
                           "terms", "cterms", "ccterms", "adjto", "adjto.data.frame",
                           "model.frame"
                         ),
                         se.fit = FALSE, codes = FALSE,
                         posterior.summary = c("mean", "median", "all"), cint = 0.95) {
  qyde <- object$qyde
  if (!length(qyde)) qyde <- 0L
  if (qyde == 0) {
    return(predict.blrm(object, ...,
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

##' Function Generator for Mean Y for `borm()`
##'
##' Delegates entirely to [Mean.blrm()], which is already link-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`). See [Mean.blrm()] for details. Not yet implemented when the fit used `yde`/`ytrans` (errors informatively rather than silently ignoring the Y-dependent effect, since [Mean.blrm()]'s own logic is hardcoded to a single-column `cppo`).
##' @param object a [borm()] fit object
##' @param ... passed to [Mean.blrm()]
##' @return see [Mean.blrm()]
##' @seealso [Mean.blrm()]
##' @author Frank Harrell
##' @method Mean borm
##' @export
Mean.borm <- function(object, ...) {
  bormStopIfYde(object, "Mean.borm")
  Mean.blrm(object, ...)
}

##' Function Generator for Quantiles of Y for `borm()`
##'
##' Delegates entirely to [Quantile.blrm()], which is already link-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`). See [Quantile.blrm()] for details. Not yet implemented when the fit used `yde`/`ytrans` (errors informatively rather than silently ignoring the Y-dependent effect, since [Quantile.blrm()]'s own logic is hardcoded to a single-column `cppo`).
##' @param object a [borm()] fit object
##' @param ... passed to [Quantile.blrm()]
##' @return see [Quantile.blrm()]
##' @seealso [Quantile.blrm()]
##' @author Frank Harrell
##' @method Quantile borm
##' @export
Quantile.borm <- function(object, ...) {
  bormStopIfYde(object, "Quantile.borm")
  Quantile.blrm(object, ...)
}

##' Exceedance Probability Function Generator for `borm()`
##'
##' Delegates entirely to [ExProb.blrm()], which is already link-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`). See [ExProb.blrm()] for details. Not yet implemented when the fit used `yde`/`ytrans` (errors informatively rather than silently ignoring the Y-dependent effect, since [ExProb.blrm()]'s own logic is hardcoded to a single-column `cppo`).
##' @param object a [borm()] fit object
##' @param ... passed to [ExProb.blrm()]
##' @return see [ExProb.blrm()]
##' @seealso [ExProb.blrm()]
##' @author Frank Harrell
##' @method ExProb borm
##' @export
ExProb.borm <- function(object, ...) {
  bormStopIfYde(object, "ExProb.borm")
  ExProb.blrm(object, ...)
}
