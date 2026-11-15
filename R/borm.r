##' Bayesian Ordinal Regression
##'
##' Uses `rstan` with pre-compiled Stan code, or `cmdstan`, to get posterior draws of parameters from a Bayesian ordinal (or binary) cumulative probability regression model, with a choice of 5 link families -- `borm()` is to [blrm()] as [rms::orm()] is to [rms::lrm()]: it generalizes the single (logistic) link that `blrm()` implements to all of the link families `orm()` supports, and is designed, like `orm()`, to scale to a very large number of distinct Y values.  Unlike `blrm()`, `borm()` does not (yet) implement the partial proportional odds model (`ppo`/`cppo`); `blrm()` remains the function to use for that.  A future Y-dependent effect generalization (`ydep`) is planned for `borm()` and will play `cppo`'s role once implemented.
##'
##' The Stan code internally uses a Dirichlet prior on the multinomial (category) probabilities at the predictor means, inducing a prior for the intercepts, exactly as [blrm()]'s own default (`iprior=0`) does -- `borm()` does not offer `blrm()`'s other two intercept-prior choices, since simulations showed they slow sampling badly once the number of distinct Y values is large, the regime `borm()` is specifically meant for.  Likewise, no data-dependent conditionals appear in the Stan code at all: censoring is handled entirely through subscripts precomputed once in R by [rms::ormIdx()] (the same routine [rms::orm.fit()] uses), not through per-observation branching in Stan.
##'
##' Design matrix columns are centered and QR-orthogonalized (via `rmsb`'s internal `selectedQr()`, the identical mechanism [blrm()] uses) before being handed to the Stan code, so that highly collinear columns do not hinder posterior sampling; **this transformation is done entirely in R, not in the Stan code**, which receives and fits `beta` on the already-transformed scale and knows nothing of the original covariate scale. Parameters are transformed back to the original data scale before being returned. Stan diagnostic output (e.g. from [stanDx()] or [stanGet()]) will show intercepts and coefficients on the centered/QR-transformed scale; everything `borm()` itself returns and prints is on the original, uncentered scale.
##'
##' As with [blrm()], priors on regression coefficients are specified through contrasts, via `pcontrast`, rather than directly; see [blrm()]'s own documentation for the full `pcontrast` syntax, which `borm()` reuses unchanged. `npcontrast` (priors for partial-PO/`ydep` effects) is accepted as an argument for forward API compatibility but cannot actually be used yet, since `borm()` has no non-proportional-odds submodel until `ydep` is implemented; supplying it is an error.
##'
##' `borm()` implements a single-level hierarchical random intercept for repeated-measures/clustered data via `cluster()` in the model formula, exactly as [blrm()] does, and additionally implements the dual-sigma `mix_re` extension to that random intercept (from `orm.rfit()`, `rms`'s internal, undocumented random-intercept fitter) via a `mix_re()` term in the formula (used together with `cluster()`), exactly as [rms::orm()] does -- see [rms::orm.fit()]'s own documentation, under `mre`, for the full statistical description of this extension. `mix_re` is not a `borm()` argument; it is extracted from the formula the same way `cluster` is.
##'
##' When using the `cmdstan` backend, `cmdstanr` will need to compile the Stan code once per computer, only recompiling when the Stan source code changes. By default the compiled code is stored in directory `.rmsb` under your home directory. Specify `options(rmsbdir=)` to specify a different location.
##'
##' If you want to run MCMC sampling even when no inputs or Stan code have changed, i.e., to use a different random number seed for the sampling process when you did not specify `sampling.args(seed=...)`, remove the `file` before running `borm`.
##'
##' Set `options(rmsbmsg=FALSE)` to suppress certain information messages.
##' @param formula an R formula object that can use `rms` package enhancements such as the restricted interaction operator, and may include `cluster()` and (together with `cluster()`) `mix_re()` terms
##' @param family a character value specifying the distribution family, corresponding to logistic (the default), probit, loglog (`exp(-exp(-x))`; extreme value type I/Gumbel maximum), cloglog (`1-exp(-exp(x))`; Gumbel minimum), and cauchit (Cauchy) link functions for `Prob[Y >= y | X]`, matching [rms::orm()]'s own `family` argument exactly
##' @param data a data frame; defaults to using objects from the calling environment
##' @param subset a logical vector or integer subscript vector specifying which subset of data should be used
##' @param na.action default is `na.delete` to remove missings and report on them
##' @param conc the Dirichlet distribution concentration parameter for the prior distribution of cell probabilities at covariate means, with the same default as [blrm()]'s own `conc` argument: the reciprocal of 0.8 + 0.35 max(k, 3) where k is the number of Y categories. For optimizing, the concentration parameter is always 1.0, to obtain results close to the MLE for providing the posterior mode.
##' @param psigma defaults to 1 for a half-t distribution with 4 d.f., location parameter `rsdmean` and scale parameter `rsdsd`, for the random-intercept SD (or, when `mix_re` is used, for `sigma1`). Set `psigma=2` to use the exponential distribution instead.
##' @param rsdmean the assumed mean of the prior distribution of the random-intercept SD (or of `sigma1` when `mix_re` is used). When `psigma=2` this is the mean of an exponential distribution and defaults to 1. When `psigma=1` this is the mean of the half-t distribution and defaults to zero.
##' @param rsdsd applies only to `psigma=1` and is the scale parameter for the half-t distribution for the random-intercept SD (or `sigma1`), defaulting to 1. When `mix_re` is used, `rsdsd` is also reused, regardless of `psigma`, as the scale of a `normal(0, rsdsd)` prior for `sigma2` (which, unlike `sigma1`, may be negative) -- a reasonable default rather than something `orm.rfit()` (rms internal) itself specifies, since that function is a pure MLE with no prior on `sigma2` at all.
##' @param pcontrast a list specifying contrasts that are to be given Gaussian prior distributions, with exactly the syntax and semantics of [blrm()]'s own `pcontrast` argument; see its documentation for full details and examples.
##' @param npcontrast accepted for forward compatibility with a future `ydep` non-proportional-odds submodel; `borm()` has no such submodel yet, so supplying `npcontrast` is an error.
##' @param backend set to `cmdstan` to use `cmdstan` through the R `cmdstanr` package instead of the default `rstan`. You can also specify this with a global option `rmsb.backend`.
##' @param iter number of posterior samples per chain for [rstan::sampling()] to run, counting warmups
##' @param warmup number of warmup iterations to discard. Default is `iter`/2.
##' @param chains number of separate chains to run
##' @param refresh see [rstan::sampling()] and [cmdstanr::sample()]. The default is 0, indicating that no progress notes are output. If `refresh > 0` and `progress` is not `''`, progress output will be appended to file `progress`.
##' @param progress see `refresh`. Defaults to `''` if `refresh = 0`.
##' @param x set to `FALSE` to not store the design matrix in the fit. `x=TRUE` is needed if running `bormStats` for example.
##' @param y set to `FALSE` to not store the response variable in the fit
##' @param loo set to `FALSE` to not run `loo` and store its result as object `loo` in the returned object. `loo` defaults to `FALSE` if the sample size is greater than 1000.
##' @param ppairs set to a file name to run `rstan` `pairs` or, if `backend='cmdstan'` `bayesplot::mcmc_pairs` and store the resulting png plot there. Set to `TRUE` instead to directly plot these diagnostics.
##' @param method set to `'optimizing'` to run the Stan optimizer and not do posterior sampling, `'both'` (the default) to run both the optimizer and posterior sampling, or `'sampling'` to run only the posterior sampling. Running `optimizing` is a way to obtain maximum likelihood estimates. When `cluster()` is present in the formula, `method` is automatically set to `'sampling'` as maximum likelihood estimates without marginalizing over the random effects do not make sense.
##' @param inito initial value for optimization. The default is the `rstan` default `'random'`. Frequently specifying `inito=0` will benefit when the number of distinct Y categories grows.
##' @param inits initial value for sampling, defaults to `inito`
##' @param standata set to `TRUE` to return the Stan data list and not run the model
##' @param debug set to `TRUE` to output timing and progress information to /tmp/debug.txt
##' @param file a file name for a `saveRDS`-created file containing or to contain the saved fit object. If `file` is specified and the file does not exist, it will be created right before the fit object is returned, less the large `rstan` object. If the file already exists and its inputs are unchanged, the previously stored fit object is immediately returned and no new calculations are done.
##' @param sampling.args a list containing parameters to pass to [rstan::sampling()] or to the `cmdstanr` `sample` function, other than `iter, warmup, chains, refresh, init`, which are already arguments to `borm`. A good use of this is `sampling.args=list(seed=3)` to get reproducible sampling.
##' @param showopt set to `TRUE` to show Stan optimizer output
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
borm <- function(formula, family = c("logistic", "probit", "loglog", "cloglog", "cauchit"),
                 data = environment(formula), subset, na.action = na.delete,
                 conc = 1. / (0.8 + 0.35 * max(k, 3)),
                 psigma = 1, rsdmean = if (psigma == 1) 0 else 1,
                 rsdsd = 1, pcontrast = NULL, npcontrast = NULL,
                 backend = c("rstan", "cmdstan"),
                 iter = 2000, warmup = iter / 2, chains = 4, refresh = 0,
                 progress = if (refresh > 0) "stan-progress.txt" else "",
                 x = TRUE, y = TRUE, loo = n <= 1000, ppairs = NULL,
                 method = c("both", "sampling", "optimizing"),
                 inito = "random", inits = inito,
                 standata = FALSE, file = NULL, debug = FALSE,
                 sampling.args = NULL, showopt = FALSE, ...) {
  call <- match.call()
  method <- match.arg(method)
  family <- match.arg(family)
  if (length(npcontrast)) {
    stop("npcontrast cannot be used until ydep is implemented in borm(); ",
         "reserved for future use")
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

  data <- rms::modelData(data, formula,
    subset = subset,
    na.action = na.action, callenv = callenv
  )

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

  wqrX <- selectedQr(X, center = TRUE)
  Xs <- trimit(wqrX$X)
  xbar <- wqrX$xbar
  xqrsd <- apply(Xs, 2, sd)
  wqrX$X <- NULL

  n <- nrow(X)
  p <- ncol(X)

  ylev <- ay$levels
  mediany <- ay$median
  midy <- ay$mid
  numy <- ay$freq
  names(numy) <- ylev

  nrp <- length(ylev) - 1L
  kmid <- if (nrp == 1) {
    1L
  } else if (length(mediany)) {
    max(1, which(ylev == mediany) - 1L)
  } else {
    max(1, which((1L:length(ylev)) == mediany) - 1L)
  }

  ass <- rms::DesignAssign(atr, nrp, Terms)

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
    lp2 = rep(0., n),
    ia = idx$ia, ia2 = idx$ia2, sgn = idx$sgn,
    conc = conc,
    cn = 0L, C = array(0., c(0, p)), cmus = array(numeric(0)), csds = array(numeric(0))
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
    yname = yname, ylevels = ylev, freq = numy,
    pppo = 0L, cppo = NULL, # degrades blrm()-shared predict/Mean/Quantile/ExProb code correctly
    Design = atr, zDesign = NULL,
    scale.pred = switch(family,
      logistic = c("log odds", "Odds Ratio"),
      loglog   = c("log hazard", "Hazard Ratio"),
      c(family, "")
    ),
    terms = Terms, assign = ass, zassign = NULL,
    na.action = atrx$na.action, fail = FALSE,
    non.slopes = nrp, interceptRef = kmid,
    sformula = sformula, zsformula = NULL,
    pcontrast = pcontrast, npcontrast = NULL
  )

  fo <- fit.object

  if (length(pcontrast)) {
    w <- docon(fo, X, wqrX, pcontrast, main = TRUE, ylev = ylev, pppo = 0L,
               clas = c("borm", "rmsb", "rms"))
    fit.object$Contrast <- w$Contrast
    fit.object$Contrastt <- w$Contrastt
    d$cn <- w$cn
    d$C <- w$C
    d$cmus <- w$cmus
    d$csds <- w$csds
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
      names(alphas) <- if (nrp == 1) "Intercept" else paste0("y>=", ylev[-1])
      opt <- list(
        coefficients = c(alphas, betas),
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
  incpars <- c("alpha", "beta", if (length(cluster)) c("gamma", sigmaname),
               if (backend == "rstan" & loo) "log_lik")

  stime <- system.time({
    args <- switch(backend,
      rstan = c(
        list(mod,
          pars = incpars, data = d, iter = iter,
          warmup = warmup, chains = chains,
          refresh = refresh
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

  alphacorr <- rowSums(sweep(betas, 2, xbar, "*"))
  alphas <- sweep(alphas, 1, alphacorr, "-")

  colnames(alphas) <- if (nrp == 1) "Intercept" else paste0("y>=", ylev[-1])
  colnames(betas) <- atr$colnames

  draws <- cbind(alphas, betas)

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
          rstan = rstan::loo(g),
          cmdstan = g$loo()
        ),
        error = function(...) list(fail = TRUE)
      )
    )
    if (itfailed(Loo)) {
      warning("loo failed; try running on loo(stanGet(fit object)) for more information")
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
      xbar = xbar, zbar = NULL,
      x = if (x) X, y = if (y) Y,
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

##' Compute Indexes of Predictive Accuracy and Their Uncertainties for `borm()`
##'
##' For an ordinal or binary regression fit from [borm()], computes several indexes of predictive accuracy along with highest posterior density intervals for them, closely paralleling [blrmStats()] (see its documentation for `g`, `gp`, `B`, `EV`, `v`, `vp`), but generalized two ways: the predicted-probability-scale quantities (`gp`, `B`, `EV`, `vp`) use the actual link family's cumulative distribution function (via `rms::probabilityFamilies`) rather than assuming logistic, and `Dxy`/`C` and the additional `rho` (Spearman's rank correlation between the linear predictor and Y) follow [rms::orm()]'s own censoring rules exactly: `rho` is `NA` whenever any censoring is present; `Dxy`/`C` are `NA` only when left- or interval-censoring is present (right-censoring alone is handled via [rms::Ocens2Surv()] and [survival::concordancefit()], the same mechanism [rms::orm.fit()] uses).
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

  f <- fit[c("x", "y", "non.slopes", "interceptRef", "draws", "ylevels", "Ncens1", "link")]
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
  s <- draws[, c(kmid, (nrp + 1) : ncol(draws)), drop = FALSE]
  ndraws <- nrow(s)
  ns <- min(ndraws, ns)
  if (ns < ndraws) {
    j <- sample(1:ndraws, ns, replace = FALSE)
    s <- s[j, , drop = FALSE]
  }
  ylev <- f$ylevels
  cumprob <- eval(rms::probabilityFamilies[[f$link]][1])

  stats <- matrix(NA, nrow = ns, ncol = 9)
  colnames(stats) <- c("Dxy", "C", "rho", "g", "gp", "B", "EV", "v", "vp")
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
      br2(prb), var(lp), var(prb)
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
  Ncens <- x$Ncens1
  L <- if (Ncens[1] > 0) paste0("L=", Ncens[1])
  R <- if (Ncens[2] > 0) paste0("R=", Ncens[2])
  int <- if (Ncens[3] > 0) paste0("I=", Ncens[3])
  ce <- if (sum(Ncens) > 0) sum(Ncens)
  ced <- if (sum(Ncens) > 0) {
    paste0(paste(c(L, R, int), collapse = ", "))
  }
  stime <- if (length(x$sampling_time)) paste0(round(x$sampling_time, 1), "s")

  misc <- rms::reListclean(
    Obs = x$N,
    Censored = ce,
    " " = ced,
    Draws = nrow(x$draws),
    Chains = x$chains,
    Time = stime,
    p = x$p,
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
    g = a["g"],
    gp = a["gp"],
    EV = a["EV"],
    v = a["v"],
    vp = a["vp"]
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
##' Predict method for [borm()] objects; delegates entirely to [predict.blrm()], which is already link-and-model-structure-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`, with its only `ppo`/`cppo`-specific logic guarded behind fields `borm()` always sets to 0/`NULL`). See [predict.blrm()] for the full set of arguments and their meaning.
##' @param object a [borm()] fit object
##' @param ... passed to [predict.blrm()]
##' @return see [predict.blrm()]
##' @seealso [predict.blrm()]
##' @author Frank Harrell
##' @method predict borm
##' @export
predict.borm <- function(object, ...) predict.blrm(object, ...)

##' Function Generator for Mean Y for `borm()`
##'
##' Delegates entirely to [Mean.blrm()], which is already link-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`). See [Mean.blrm()] for details.
##' @param object a [borm()] fit object
##' @param ... passed to [Mean.blrm()]
##' @return see [Mean.blrm()]
##' @seealso [Mean.blrm()]
##' @author Frank Harrell
##' @method Mean borm
##' @export
Mean.borm <- function(object, ...) Mean.blrm(object, ...)

##' Function Generator for Quantiles of Y for `borm()`
##'
##' Delegates entirely to [Quantile.blrm()], which is already link-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`). See [Quantile.blrm()] for details.
##' @param object a [borm()] fit object
##' @param ... passed to [Quantile.blrm()]
##' @return see [Quantile.blrm()]
##' @seealso [Quantile.blrm()]
##' @author Frank Harrell
##' @method Quantile borm
##' @export
Quantile.borm <- function(object, ...) Quantile.blrm(object, ...)

##' Exceedance Probability Function Generator for `borm()`
##'
##' Delegates entirely to [ExProb.blrm()], which is already link-agnostic (dispatches via `object$link`/`rms::probabilityFamilies`). See [ExProb.blrm()] for details.
##' @param object a [borm()] fit object
##' @param ... passed to [ExProb.blrm()]
##' @return see [ExProb.blrm()]
##' @seealso [ExProb.blrm()]
##' @author Frank Harrell
##' @method ExProb borm
##' @export
ExProb.borm <- function(object, ...) ExProb.blrm(object, ...)
