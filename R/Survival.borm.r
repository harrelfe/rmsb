# Claude Sonnet 5 2026-11-22
# `time` (used below, inside Survival.borm()'s returned closure) is a
# column of the local data frame `w`, used through subset()'s NSE --
# the same idiom rms::ormYdeSurvival() itself uses (predict.lrm.s:695)
# -- which R CMD check's static analysis cannot tell from a real
# global variable; rms's own existing convention for this exact false
# positive (see e.g. R/stanMisc.r's two utils::globalVariables()
# calls) is followed here. Placed before the roxygen block (rather
# than directly above the function it concerns) because roxygen2's
# @method/@export handling for Survival.borm requires its own comment
# block to be immediately adjacent to the function assignment --
# confirmed directly: inserting this call between them silently
# dropped the S3method(Survival, borm) registration from NAMESPACE.
utils::globalVariables('time')

##' Function Generator for Survival Probabilities for a `borm()` Fit
##'
##' Creates a function to compute survival probabilities (`P(T>t)`) for a
##' [borm()] fit, fully accounting for posterior uncertainty: every
##' derived quantity (the survival probability itself, and its highest
##' posterior density interval when requested) is computed separately for
##' each posterior draw, then summarized across draws -- never a single
##' posterior-summarized coefficient vector applied once (the simpler,
##' but less correct for a nonlinear quantity, approach [Mean.blrm()] and
##' its relatives use). This parallels `rms::ormYdeSurvival()`, generalized
##' to borm's one new dimension (posterior draws) the same way
##' `bormYdeMean()`/`bormYdeQuantile()`/`bormYdeExProb()` already
##' generalize their own `rms::ormYde*()` counterparts -- a parallel
##' implementation, not a shared call across the package boundary, per
##' the same modularization decision as those three.
##'
##' Requires `y=TRUE` to have been given to [borm()]. Unlike
##' `rms::Survival.orm()`, which stores its own censoring metadata
##' (`tail`/`ranges`/`yupper`) as top-level fit-object fields regardless
##' of `y=`, `borm()` does not promote them -- confirmed directly, by
##' fitting censored models both ways, that they are still fully present
##' as `attributes(object$y)$tail`/`$ranges`/`$upper`, with identical
##' values to `rms::orm()`'s own `object$tail`/`$ranges`/`$yupper` on the
##' same data, but only reachable that way.
##'
##' The one new efficiency lever posterior draws require: when the
##' caller gives `times=` as a modest, explicit set of landmark times,
##' only the (typically few) distinct intercepts those times actually map
##' to are ever extracted from `object$draws` or run through the
##' per-draw loop -- never all `k` of them, however large `k` is. When
##' `times=` is omitted (the default: the curve at every modeled
##' intercept, as `rms::Survival.orm()`'s own default also does), that
##' efficiency lever is unavailable by construction -- the caller is
##' asking for every intercept. A point estimate (`cint=0`, or
##' `posterior.summary='mean'` with no `cint`) stays cheap regardless,
##' via a running sum across draws that is never fully materialized
##' (`O(n times the number of distinct intercepts touched)` memory,
##' independent of the number of posterior draws). A highest posterior
##' density interval, or `posterior.summary='median'`, cannot be
##' streamed this way -- both need every draw's value at every touched
##' point stored at once. When that would require more than
##' `hpd_threshold` total values, this stops with an informative error
##' (naming the actual counts) rather than attempting to materialize an
##' array that may not fit in memory -- Frank's explicit choice, with
##' `hpd_threshold=1.5e7` as the suggested default.
##' @param object a [borm()] fit object, fitted with `y=TRUE`
##' @param precision number of decimal places to round `times` to before
##'   deduplicating (matches `rms::Survival.orm()`'s own argument)
##' @param hpd_threshold maximum allowed `(number of posterior draws) *
##'   (number of distinct intercepts actually needed)` before a request
##'   needing every draw's value (`conf.int>0`, or
##'   `posterior.summary='median'`) is refused rather than attempted
##' @param ... ignored; present only so this method's formals match the
##'   `Survival` generic's own `function(object, ...)` signature
##' @return a function of `times`, `newdata` (required -- there is no
##'   single linear predictor that represents every posterior draw, so
##'   unlike `rms::Survival.orm()` there is no `lp=`/`X=` alternative),
##'   `posterior.summary`, `cint`, `forcedf`, `zero`, and `precision`,
##'   paralleling `rms::Survival.orm()`'s own returned function
##' @author Frank Harrell
##' @method Survival borm
##' @export
##' @md
Survival.borm <- function(object, precision = 7, hpd_threshold = 1.5e7, ...) {
  qyde <- object$qyde
  if (! length(qyde)) qyde <- 0L
  ylevels  <- object$yunique
  k        <- object$non.slopes
  cumprob  <- eval(rms::probabilityFamilies[[object$family]][1])
  draws    <- object$draws
  ndraws   <- nrow(draws)
  alphaname <- if (k == 1) "Intercept" else paste0("y>=", ylevels[-1])

  # Claude Sonnet 5 2026-11-22          entire block
  # y=TRUE is a hard requirement here, unlike rms::Survival.orm(): the
  # censoring metadata this function needs is only retained as
  # attributes(object$y)$tail/$ranges/$upper (confirmed directly, by
  # fitting left-, right-, and interval-censored borm() models and
  # comparing attributes(fit$y) to the matching orm() fit's own
  # object$tail/$ranges/$yupper -- identical values, different access
  # path), not promoted to top-level fit-object fields the way
  # rms::orm.fit() promotes them regardless of y=.
  Yobj <- object$y
  if (! length(Yobj))
    stop("Survival() for a borm() fit requires y=TRUE to have been ",
         "specified when the model was fit -- the censoring information ",
         "it needs is only retained there")
  ranges   <- attributes(Yobj)$ranges
  tailinfo <- attributes(Yobj)$tail
  yupper   <- attributes(Yobj)$upper

  vals <- as.numeric(ylevels)
  if (length(yupper))
    vals <- as.numeric(ifelse(is.infinite(vals), yupper,
                               ifelse(is.infinite(yupper), vals, (vals + yupper) / 2.)))

  if (qyde > 0) {
    ytrans    <- eval(parse(text = object$ytrans))
    ytransmat <- ytrans(ylevels)
    if (! is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)
    # Claude Sonnet 5 2026-11-23: myde is not stored -- recompute it
    myde <- ncol(ytransmat)
    # Claude Sonnet 5 2026-11-22
    # Row alignment confirmed earlier this session (bormPredictYde()'s
    # own type='fitted'/'mean' extension): borm()'s ytransmat has one
    # row per Y level INCLUDING the first (unlike rms::orm.fit()'s
    # convention), so dropping its own first row aligns row j with
    # alpha[j]/alphaname[j].
    ytransmat_cut <- ytransmat[-1, , drop = FALSE]   # k x myde
    taunam   <- bormYdeNames(object$zDesign$colnames, ytransmat, qyde, myde)
    betacols <- setdiff(colnames(draws), c(alphaname, taunam))
  } else {
    betacols <- setdiff(colnames(draws), alphaname)
  }

  g <- function(times = numeric(0), newdata = NULL,
                posterior.summary = c('mean', 'median'), cint = 0,
                forcedf = FALSE, zero = FALSE, precision = 7, ...) {
    if (! length(newdata))
      stop("Survival() for a borm() fit must be called with newdata (the ",
           "original covariate data frame) -- there is no single linear ",
           "predictor that represents every posterior draw.")
    posterior.summary <- match.arg(posterior.summary, c('mean', 'median'))

    len_t <- length(times)
    if (len_t) {
      mul   <- 10 ^ precision
      times <- round(times * mul) / mul
    }
    zero <- zero && ! len_t && (vals[1] != 0)
    if (len_t) times <- sort(unique(times))
    else {
      times <- vals
      if (length(tailinfo)) times <- vals[- tailinfo$index]
    }

    kk <- k + 1L   # total intercept "slots", including the fake trailing one

    X <- predict(object, newdata, type = 'x', Zmatrix = FALSE)
    if (! is.matrix(X)) X <- matrix(X, nrow = 1)
    n <- nrow(X)
    if (qyde > 0) Z <- bormYdeZ(object, newdata)

    xrow <- seq_len(n)
    w    <- expand.grid(time = times, Xrow = xrow)
    ts   <- w$time
    ints <- approx(vals, 1 : kk, xout = ts, method = 'constant', rule = 2)$y
    # Claude Sonnet 5 2026-11-22          entire block
    # The efficiency lever posterior draws need: only the distinct
    # intercepts actually touched by the requested/defaulted times
    # (`needed`) are ever extracted from `draws` or run through the
    # per-draw loop -- never all k of them unless every one really is
    # needed (the times=NULL/full-curve case). `needed` is sorted, so
    # the fake slot kk, if present at all, is always its last element.
    # The last REAL intercept (k) is forced into `needed` whenever
    # there is a censoring tail, regardless of whether any requested
    # time's own grid index happens to equal k -- confirmed directly
    # (by a failing test) that the tail carry-forward region
    # (max_uncens, max_fu] can be reached by times whose OWN grid
    # index is < k (the grid is built from distinct Y values, which
    # can include censored times larger than max_uncens), so without
    # this, surv_last/lastlower/lastupper could be requested (via
    # use_last below) without ever having been computed -- unlike
    # rms::ormYdeSurvival()'s own code, which always has every column
    # available (it builds the full n x L matrix) and so never faces
    # this at all.
    needed   <- sort(unique(c(ints, if (length(tailinfo)) k)))
    fake     <- needed[length(needed)] == kk
    needed_k <- if (fake) needed[- length(needed)] else needed
    pos      <- match(ints, needed)   # maps each w-row to its column within `needed`
    idx      <- cbind(w$Xrow, pos)

    needFull <- cint > 0 || posterior.summary == 'median'
    total_cells <- ndraws * length(needed)
    if (needFull && total_cells > hpd_threshold)
      stop("Survival() for this borm() fit would need every one of ", ndraws,
           " posterior draws' values at ", length(needed), " distinct times/",
           "intercepts (", format(total_cells, big.mark = ','), " total values) ",
           "to compute ", if (cint > 0) "conf.int" else "posterior.summary='median'",
           ", exceeding the limit of ", format(hpd_threshold, big.mark = ','),
           " -- specify fewer times=, or use the default posterior.summary='mean' ",
           "with cint=0 for a cheap running-mean result instead")

    # Claude Sonnet 5 2026-11-22
    # Returns an n x length(needed) matrix for one posterior draw,
    # exactly the bilinear structure (Z %*% Tau %*% t(ytransmat)) used
    # throughout this session's yde work, here applied only at the
    # columns `needed_k` actually required -- the fake slot's column
    # (survival 0, matching cumprob(-Inf)) is filled in directly, not
    # computed, since it isn't a real intercept at all.
    perDraw <- function(d) {
      beta_d  <- draws[d, betacols]
      Xbeta_d <- if (length(betacols)) as.vector(X %*% beta_d) else numeric(n)
      out <- matrix(NA_real_, n, length(needed))
      if (length(needed_k)) {
        alpha_d <- draws[d, alphaname[needed_k]]
        if (qyde > 0) {
          Tau_d  <- matrix(draws[d, taunam], nrow = qyde, ncol = myde)
          W_d    <- Z %*% Tau_d
          ZTau_d <- W_d %*% t(ytransmat_cut[needed_k, , drop = FALSE])
          lp_d   <- sweep(ZTau_d, 2, alpha_d, '+') + Xbeta_d
        } else {
          lp_d <- outer(Xbeta_d, alpha_d, '+')
        }
        out[, seq_along(needed_k)] <- cumprob(lp_d)
      }
      if (fake) out[, length(needed)] <- 0.
      out
    }

    lastpos <- match(k, needed)   # column, within `needed`, of the last REAL intercept

    if (! needFull) {
      Ssum <- matrix(0, n, length(needed))
      for (d in 1 : ndraws) Ssum <- Ssum + perDraw(d)
      Pfull <- Ssum / ndraws
      surv      <- Pfull[idx]
      surv_last <- if (!is.na(lastpos)) Pfull[, lastpos] else rep(NA_real_, n)
    } else {
      Svals <- matrix(NA_real_, nrow(idx), ndraws)
      Lvals <- if (!is.na(lastpos)) matrix(NA_real_, n, ndraws) else NULL
      for (d in 1 : ndraws) {
        Pd <- perDraw(d)
        Svals[, d] <- Pd[idx]
        if (!is.na(lastpos)) Lvals[, d] <- Pd[, lastpos]
      }
      postsum <- if (posterior.summary == 'mean') rowMeans else
        function(x) apply(x, 1, median)
      surv      <- postsum(Svals)
      surv_last <- if (!is.na(lastpos)) postsum(Lvals) else rep(NA_real_, n)
      if (cint > 0) {
        hpd <- apply(Svals, 1, HPDint, prob = cint)
        lower <- hpd[1, ]; upper <- hpd[2, ]
        if (!is.na(lastpos)) {
          hpdL <- apply(Lvals, 1, HPDint, prob = cint)
          lastlower <- hpdL[1, ]; lastupper <- hpdL[2, ]
        }
      }
    }

    # Claude Sonnet 5 2026-11-22
    # Boundary/censoring-tail decisions are identical to
    # rms::ormYdeSurvival()'s own (non-Bayesian) code -- only the
    # *values* being overridden differ (a posterior mean/median, or
    # also an HPD interval, instead of a single cump(lpsi)).
    min_uncens <- ranges$u[1]
    max_uncens <- ranges$u[2]
    surv[ts < min_uncens] <- 1.0
    haveCI <- cint > 0
    if (haveCI) { lower[ts < min_uncens] <- 1.0; upper[ts < min_uncens] <- 1.0 }
    if (length(tailinfo)) {
      max_fu   <- tailinfo$range[2]
      use_last <- ts > max_uncens & ts <= max_fu
      surv[ts > max_fu] <- NA
      surv[use_last] <- surv_last[w$Xrow[use_last]]
      if (haveCI) {
        lower[ts > max_fu] <- NA; upper[ts > max_fu] <- NA
        lower[use_last] <- lastlower[w$Xrow[use_last]]
        upper[use_last] <- lastupper[w$Xrow[use_last]]
      }
    } else {
      surv[ts >= max_uncens] <- 0.0
      if (haveCI) { lower[ts >= max_uncens] <- 0.0; upper[ts >= max_uncens] <- 0.0 }
    }

    w <- data.frame(time = ts, surv, Xrow = w$Xrow)
    if (haveCI) { w$lower <- lower; w$upper <- upper }
    if (n == 1) w$Xrow <- NULL
    row.names(w) <- NULL
    if (zero) {
      z <- subset(w, time == min(w$time))
      z$time <- 0; z$surv <- 1
      if (haveCI) { z$lower <- 1; z$upper <- 1 }
      w <- rbind(z, w)
      row.names(w) <- NULL
    }
    if (forcedf) w
    else if (len_t == 1 && ! haveCI) w$surv
    else if (n == 1 && ! haveCI) structure(w$surv, names = paste0('P(T>', w$time, ')'))
    else w
  }
  formals(g)$precision <- precision
  g
}

##' Survival Estimates for a `borm()` Fit
##'
##' Thin wrapper around [Survival.borm()], paralleling `rms::survest.orm()`'s
##' own role (both now read `object$yunique`, following `rms::orm()`'s
##' own field-naming convention -- Claude Sonnet 5 2026-11-23), and,
##' because there is no single linear predictor representing every
##' posterior draw, always requiring `newdata` rather than accepting a
##' bare `linear.predictors=`/`x=`.
##' @param fit a [borm()] fit object, fitted with `y=TRUE`
##' @param newdata data frame defining covariate settings (required)
##' @param times times for which estimates are desired; defaults, as in
##'   [Survival.borm()], to every modeled intercept
##' @param posterior.summary `'mean'` (the default) or `'median'`
##' @param conf.int highest posterior density interval probability;
##'   default `0.95`, set to `0` for none
##' @param ... ignored
##' @return a data frame with variables `time`, `surv`, and (if
##'   `conf.int>0`) `lower`, `upper`; `Xrow` indicates which row of
##'   `newdata` each estimate belongs to
##' @author Frank Harrell
##' @method survest borm
##' @export
##' @md
survest.borm <- function(fit, newdata = NULL, times = NULL,
                         posterior.summary = c('mean', 'median'),
                         conf.int = 0.95, ...) {
  if (! length(newdata))
    stop("survest() for a borm() fit must be called with newdata (the ",
         "original covariate data frame)")
  posterior.summary <- match.arg(posterior.summary)
  S <- Survival(fit)
  S(times, newdata = newdata, posterior.summary = posterior.summary,
    cint = conf.int, forcedf = TRUE, zero = TRUE)
}
