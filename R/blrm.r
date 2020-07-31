##' Bayesian Binary and Ordinal Logistic Regression
##'
##' Uses `rstan` with pre-compiled Stan code to get posterior draws of parameters from a binary logistic or proportional odds semiparametric ordinal logistic model.  The Stan code internally using the qr decompositon on the design matrix so that highly collinear columns of the matrix do not hinder the posterior sampling.  The parameters are transformed back to the original scale before returning results to R.   Design matrix columns are centered before running Stan, so Stan diagnostic output will have the intercept terms shifted but the results of [blrm()] for intercepts are for the original uncentered data.  The only prior distributions for regression betas are normal with mean zero, and the vector of prior standard deviations is given in `priorsd`.  These priors are for the qr-projected design matrix elements, except that the very last element is not changed.  So if one has a single non-interactive linear or binary variable for which a skeptical prior is designed, put that variable last in the model.
##'
##' The partial proportional odds model of Peterson and Harrell (1990) is implemented, and is invoked when the user specifies a second model formula as the `ppo` argument.  This formula has no left-hand-side variable, and has right-side variables that are a subset of those in `formula` specifying for which predictors the proportional odds assumption is relaxed.
##'
##' The Peterson and Harrell (1990) constrained partial proportional odds is also implemented, and is usually preferred to the above unconstrained PPO model as it adds a vector of coefficients instead of a matrix of coefficients.  In the constrained PPO model the user provides a function `cppo` that computes a score for all observed values of the dependent variable.  For example with a discrete ordinal outcome `cppo` may return a value of 1.0 for a specific value of Y and zero otherwise.  That will result in a departure from the proportional odds assumption for just that one level of Y.  The value returned by `cppo` at the lowest Y value is never used in any case.
##'
##' [blrm()] also handles single-level hierarchical random effects models for the case when there are repeated measurements per subject which are reflected as random intercepts, and a different experimental model that allows for AR(1) serial correlation within subject.  For both setups, a `cluster` term in the model signals the existence of subject-specific random effects.
##'
##' See <https://hbiostat.org/R/rms/blrm.html> for multiple examples with results.
##' @title blrm
##' @param formula a R formula object that can use `rms` package enhancements such as the restricted interaction operator
##' @param ppo formula specifying the model predictors for which proportional odds is not assumed
##' @param cppo a function that if present causes a constrained partial PO model to be fit.  The function specifies the values in the Gamma vector in Peterson and Harrell (1990) equation (6).  To make posterior sampling better behaved, the function should be scaled and centered.  This is done by wrapping `cppo` in a function that scales the `cppo` result before return the vector value.  See the `normcco` argument for how to prevent this.  The default normalization is based on the mean and standard deviation of the function values over the distribution of observed Y.  For getting predicted values and estimates post-[blrm()], `cppo` must not reference any functions that are not available at such later times.
##' @param keepsep a single character string containing a regular expression applied to design matrix column names, specifying which columns are not to be QR-orthonormalized, so that priors for those columns apply to the original parameters.  This is useful for treatment and treatment interaction terms.  For example `keepsep='treat'` will keep separate all design matrix columns containing `'treat'` in their names.  Some characters such as the caret used in polynomial regression terms will need to be escaped by a double backslash.
##' @param data a data frame; defaults to using objects from the calling environment
##' @param subset a logical vector or integer subscript vector specifying which subset of data whould be used
##' @param na.action default is `na.delete` to remove missings and report on them
##' @param priorsd vector of prior standard deviations.  If the vector is shorter than the number of model parameters, it will be repeated until the length equals the number of parametertimes.
##' @param priorsdppo vector of prior standard deviations for non-proportional odds parameters.  As with `priorsd` the last element is the only one for which the SD corresponds to the original data scale.
##' @param conc the Dirichlet distribution concentration parameter for the prior distribution of cell probabilities at covariate means.  The default is the reciprocal of 0.8 + 0.35 max(k, 3) where k is the number of Y categories.  The default is chosen to make the posterior mean of the intercepts more closely match the MLE.  For optimizing, the concentration parameter is always 1.0 to obtain results very close to the MLE for providing the posterior mode.
##' @param psigma defaults to 1 for a half-t distribution with 4 d.f., location parameter `rsdmean` and scale parameter `rsdsd`.  Set `psigma=2` to use the exponential distribution.
##' @param rsdmean the assumed mean of the prior distribution of the standard deviation of random effects.  When `psigma=2` this is the mean of an exponential distribution and defaults to 1.  When `psigma=1` this is the mean of the half-t distribution and defaults to zero.
##' @param rsdsd applies only to `psigma=1` and is the scale parameter for the half t distribution for the SD of random effects, defaulting to 1.
##' @param normcppo set to `FALSE` to leave the `cppo` function as-is without automatically centering and scaling the result
##' @param iter number of posterior samples per chain for [rstan::sampling()] to run
##' @param chains number of separate chains to run
##' @param refresh see [rstan::sampling()].  The default is 0, indicating that no progress notes are output.  If `refresh > 0` and `progress` is not `''`, progress output will be appended to file `progress`.  The default file name is `'stan-progress.txt'`.
##' @param progress see `refresh`.  Defaults to `''` if `refresh = 0`.  Note: If running interactively but not under RStudio, `rstan` will open a browser window for monitoring progress.
##' @param x set to `FALSE` to not store the design matrix in the fit.  `x=TRUE` is needed if running `blrmStats` for example.
##' @param y set to `FALSE` to not store the response variable in the fit
##' @param loo set to `FALSE` to not run `loo` and store its result as object `loo` in the returned object.  `loo` defaults to `FALSE` if the sample size is greater than 1000, as `loo` requires the per-observation likelihood components, which creates a matrix N times the number of posterior draws.
##' @param ppairs set to a file name to run `rstan` `pairs` and store the resulting png plot there.  Set to `TRUE` instead to directly plot these diagnostics.  The default is not to run `pairs`.
##' @param method set to `'optimizing'` to run the Stan optimizer and not do posterior sampling, `'both'` (the default) to run both the optimizer and posterior sampling, or `'sampling'` to run only the posterior sampling and not compute posterior modes. Running `optimizing` is a way to obtain maximum likelihood estimates and allows one to quickly study the effect of changing the prior distributions.  When `method='optimizing'` is used the result returned is not a standard [blrm()] object but is instead the parameter estimates, -2 log likelihood, and optionally the Hession matrix (if you specify `hessian=TRUE` in ...).  When `method='both'` is used, [rstan::sampling()] and [rstan::optimizing()] are both run, and parameter estimates (posterior modes) from `optimizing` are stored in a matrix `param` in the fit object, which also contains the posterior means and medians, and other results from `optimizing` are stored in object `opt` in the [blrm()] fit object.  When random effects are present, `method` is automatically set to `'sampling'` as maximum likelihood estimates without marginalizing over the random effects do not make sense.
##' @param inito intial value for optimization.  The default is the `rstan` default `'random'`.  Frequently specifying `init=0` will benefit when the number of distinct Y categories grows or when using `ppo` hence 0 is the default for that.
##' @param inits initial value for sampling, defaults to `inito`
##' @param standata set to `TRUE` to return the Stan data list and not run the model
##' @param debug set to `TRUE` to output timing and progress information to /tmp/debug.txt
##' @param file a file name for a `saveRDS`-created file containing or to contain the saved fit object.  If `file` is specified and the file does not exist, it will be created right before the fit object is returned, less the large `rstan` object.  If the file already exists, its stored `md5` hash string `datahash` fit object component is retrieved and compared to that of the current `rstan` inputs.  If the data to be sent to `rstan`, the priors, and all sampling and optimization options and stan code are identical, the previously stored fit object is immediately returned and no new calculatons are done.
##' @param ... passed to [rstan::optimizing()].  The `seed` parameter is a popular example.
##' @return an `rms` fit object of class `blrm`, `rmsb`, `rms` that also contains `rstan` results under the name `rstan`.  In the `rstan` results, which are also used to produce diagnostics, the intercepts are shifted because of the centering of columns of the design matrix done by [blrm()].  With `method='optimizing'` a class-less list is return with these elements: `coefficients` (MLEs), `beta` (non-intercept parameters on the QR decomposition scale), `deviance` (-2 log likelihood), `return_code` (see [rstan::optimizing()]), and, if you specified `hessian=TRUE` to [blrm()], the Hessian matrix.  To learn about the scaling of orthogonalized QR design matrix columns, look at the `xqrsd` object in the returned object.  This is the vector of SDs for all the columns of the transformed matrix.  Those kept out by the `keepsep` argument will have their original SDs.
##' @examples
##' \dontrun{
##'   getHdata(Titanic3)
##'   dd <- datadist(titanic3); options(datadist='dd')
##'   f <- blrm(survived ~ (rcs(age, 5) + sex + pclass)^2, data=titanic3)
##'   f                   # model summary using print.blrm
##'   coef(f)             # compute posterior mean parameter values
##'   coef(f, 'median')   # compute posterior median values
##'   stanDx(f)           # print basic Stan diagnostics
##'   s <- stanGet(f)     # extract rstan object from fit
##'   plot(s, pars=f$betas)       # Stan posteriors for beta parameters
##'   stanDxplot(s)       # Stan diagnostic plots by chain
##'   blrmStats(f)        # more details about predictive accuracy measures
##'   ggplot(Predict(...))   # standard rms output
##'   summary(f, ...)     # invokes summary.rms
##'   contrast(f, ...)    # contrast.rms computes HPD intervals
##'   plot(nomogram(f, ...)) # plot nomogram using posterior mean parameters
##'
##'   # Fit a random effects model to handle multiple observations per
##'   # subject ID
##'   f <- blrm(outcome ~ rcs(age, 5) + sex + cluster(id), data=mydata)
##' }
##' @author Frank Harrell and Ben Goodrich
##' @seealso [print.blrm()], [blrmStats()], [stanDx()], [stanGet()], [coef.rmsb()], [vcov.rmsb()], [print.rmsb()], [coef.rmsb()]
##' @export
blrm <- function(formula, ppo=NULL, cppo=NULL, keepsep=NULL,
                 data=environment(formula), subset, na.action=na.delete,
								 priorsd=rep(100, p), priorsdppo=rep(100, pppo),
                 conc=1./(0.8 + 0.35 * max(k, 3)),
                 psigma=1, rsdmean=if(psigma == 1) 0 else 1,
                 rsdsd=1, normcppo=TRUE,
								 iter=2000, chains=4, refresh=0,
                 progress=if(refresh > 0) 'stan-progress.txt' else '',
								 x=TRUE, y=TRUE, loo=n <= 1000, ppairs=NULL,
                 method=c('both', 'sampling', 'optimizing'),
                 inito=if(length(ppo)) 0 else 'random', inits=inito,
                 standata=FALSE, file=NULL, debug=FALSE,
                 ...) {

  call   <- match.call()
  method <- match.arg(method)

  ## Use modelData because model.frame did not work
  ## correctly when called the second time for ppo, and because of
  ## problems evaluating subset

  callenv <- parent.frame()   # don't delay these evaluations
  subset  <- if(! missing(subset )) eval(substitute(subset),  data, callenv)

  data <-
    rms::modelData(data, formula, ppo,
              subset = subset,
              na.action=na.action, callenv=callenv)
    if(length(ppo)) {
      data2 <- attr(data, 'data2')
      attr(data, 'data2') <- NULL
      }

  prevhash <- NULL
  if(length(file) && file.exists(file)) {
    prevfit  <- readRDS(file)
    prevhash <- prevfit$datahash
    }

  if(debug) debug <- function(...)
    cat(..., format(Sys.time()), '\n', file='/tmp/debug.txt', append=TRUE)
  else debug <- function(...) {}

  ## Function to drop only the ith dimension of a 3-array when subscripting
  ## that dimension to the jth element where j is a single integer
  ## result is a matrix
  ## Note: a[,,j, drop=TRUE] drops dimensions other than the 3rd
  ## when they have only one element
  dropadim <- function(a, i, j) {
    n <- dimnames(a)
    d <- dim(a)
    b <- if(i == 1)      a[j,  ,  , drop=FALSE]
         else if(i == 2) a[ , j,  , drop=FALSE]
         else            a[,   , j, drop=FALSE]
    d <- d[-i]
    dn <- if(length(n)) n[-i]
    matrix(as.vector(b), nrow=d[1], ncol=d[2], dimnames=dn)
    }

  nact <- NULL

  yname   <- as.character(formula[2])

  en <- environment(formula)

  requireNamespace('rstan', quietly=TRUE)

  ## Design handles cluster()
  X <- rms::Design(data, formula=formula, specials='cluster')

  cluster     <- attr(X, 'cluster')
  clustername <- attr(X, 'clustername')

  atrx       <- attributes(X)
  sformula   <- atrx$sformula
  nact       <- atrx$na.action
  Terms      <- atrx$terms
  attr(Terms, "formula") <- formula
  atr        <- atrx$Design
  mmcolnames <- atr$mmcolnames

  Y       <- model.extract(X, 'response')
  isOcens <- inherits(Y, 'Ocens')
  Ncens   <- c(left=0, right=0, interval=0)
  if(! isOcens) {
    Y  <- Ocens(Y)
    ay <- attributes(Y)
    k  <- length(ay$levels)
  } else {
    ay <- attributes(Y)
    k  <- length(ay$levels)
    a  <- Y[, 1]
    b  <- Y[, 2]
    Ncens <- c(left     = sum(a == 1 & b > 1),
               right    = sum(b == k & a < k),
               interval = sum(a != b & a > 1 & b < k))
  }

  offs <- atrx$offset
  if(!length(offs)) offs <- 0
  X <- model.matrix(Terms, X)
  alt <- attr(mmcolnames, 'alt')
  if(! all(mmcolnames %in% colnames(X)) && length(alt)) mmcolnames <- alt
  X <- X[, mmcolnames, drop=FALSE]
  colnames(X) <- atr$colnames
  notransX <- if(length(keepsep)) grep(keepsep, atr$colnames)
  if(length(keepsep) & ! length(notransX))
    warning('keepsep did not apply to any column of X')
  notransXn <- atr$colnames[notransX]

  Z <- notransZn <- zatr <- zsformula <- NULL

  if(length(ppo)) {
    ## Response variable is not present in data2
    Z <- rms::Design(data2, formula=ppo)

    zatrx       <- attributes(Z)
    zsformula   <- zatrx$sformula
    zTerms      <- zatrx$terms
    attr(zTerms, "formula") <- ppo
    zatr        <- zatrx$Design
    mmcolnames  <- zatr$mmcolnames

    Z <- model.matrix(zTerms, Z)
    alt <- attr(mmcolnames, 'alt')
    if(! all(mmcolnames %in% colnames(Z)) && length(alt)) mmcolnames <- alt
    Z <- Z[, mmcolnames, drop=FALSE]
    colnames(Z) <- zatr$colnames
    notransZ  <- if(length(keepsep)) grep(keepsep, zatr$colnames)
    notransZn <- zatr$colnames[notransZ]
  }

  zbar <- NULL

  wqrX  <- selectedQr(X, center=TRUE, not=notransX)
  Xs    <- wqrX$X
  xbar  <- wqrX$xbar
  xqrsd <- apply(Xs, 2, sd)
  wqrX$X <- NULL
  
  if(length(ppo)) {
    wqrZ   <- selectedQr(Z, center=TRUE, not=notransZ)
    Zs     <- wqrZ$X
    zbar   <- wqrZ$xbar
    wqrZ$X <- NULL
  }

	n    <- nrow(X)
	p    <- ncol(X)

  ylev        <- ay$levels
  mediany     <- ay$median
  midy        <- ay$mid
  numy        <- ay$freq
  names(numy) <- ylev

  kmid <- if(length(mediany))
            max(1, which(ylev == mediany) - 1L)
          else  # get intercept number closest to median Y
            max(1, which((1L : length(ylev)) == mediany) - 1L)

  pppo <- if(length(ppo)) ncol(Z) else 0
  if(length(cppo) && (pppo == 0)) stop('may not specify cppo without ppo')

  nrp               <- length(ylev) - 1L
  if(nrp == 1) kmid <- 1

	ass     <- rms::DesignAssign(atr, nrp, Terms)
  zass    <- if(pppo > 0) rms::DesignAssign(zatr, 0, zTerms)    ##  0? 1?

  priorsd <- rep(priorsd, length=p)
  ## Unconstrained PPO model does not handle censoring
  if(k > 2 && length(ppo) && ! length(cppo)) Y <- as.integer(Y[, 1])
	d <- list(X=Xs,
            y=Y,
            N=n, p=p, k=k, conc=conc,
            sds=as.array(priorsd))

  if(length(cppo)) {
    if(normcppo) {
      Cppo <- function(y, fun, center, scale) (fun(y) - center) / scale
      formals(Cppo)$fun  <- cppo
      ## Evaluate function on original scale
      cppoy <- cppo(ylev[Y[,1]])
      formals(Cppo)$center <- mean(cppoy)
      formals(Cppo)$scale  <- sd(cppoy)
      cppo <- Cppo
      }
    d$pposcore <- cppo(ylev)   # now it's centered and scaled
  } else d$pposcore <- numeric()
  d$lpposcore <- length(d$pposcore)

  Nc <- 0
  d$psigma <- as.integer(psigma)
  if(length(cluster)) {
    d$rsdmean <- as.array(rsdmean)
    d$rsdsd   <- if(psigma == 1) as.array(rsdsd) else numeric(0)
    cl        <- as.integer(as.factor(cluster))
    Nc        <- max(cl, na.rm=TRUE)
    d$Nc      <- Nc
    d$cluster <- cl
    method    <- 'sampling'
  }
  else {
    Nc <- d$Nc <- 0L
    d$rsdmean <- d$rsdsd <- numeric()
    d$cluster <- integer()
    }

  d$Z <- if(pppo) Zs else matrix(0., nrow=n, ncol=0)
  d$q <- pppo
  priorsdppo <- rep(priorsdppo, length=pppo)
  d$sdsppo   <- as.array(priorsdppo)
  
	if(any(is.na(Xs)) | any(is.na(Y))) stop('program logic error')

  unconstrainedppo <- pppo > 0 && length(cppo) == 0
  fitter <- if(unconstrainedppo) 'lrmcppo' else 'lrmconppo'
  if(unconstrainedppo) d$pposcore <- d$lpposcore <- NULL
  
  if(standata) return(d)

  mod <- stanmodels[[fitter]]
  stancode <- rstan::get_stancode(mod)

  ## See if previous fit had identical inputs and Stan code
  hashobj  <- list(d, inito, inits, iter, chains, loo, ppairs, method,
                   stancode, ...)
  datahash <- digest::digest(hashobj)
  if(length(prevhash) && prevhash == datahash) return(prevfit)

  hashobj  <- NULL

  itfailed <- function(w) is.list(w) && length(w$fail) && w$fail

  opt <- parm <- taus <- NULL

  sigmagname <- 'sigmag[1]'

  if(method != 'sampling') {
    # Temporarily make concentration parameter = 1.0
    d$conc <- 1.0

    g <- rstan::optimizing(mod, data=d, init=inito)
    d$conc <- conc   # restore
    if(g$return_code != 0) {
      warning(paste('Optimizing did not work; return code', g$return_code,
                    '\nPosterior modes not computed'))
      opt <- list(coefficients=NULL,
                  sigmag=NA, deviance=NA,
                  return_code=g$return_code, hessian=NULL)
    } else {
      parm   <- g$par
      nam    <- names(parm)
      al     <- nam[grep('alpha\\[', nam)]
      be     <- nam[grep('beta\\[',  nam)]
      ta     <- nam[grep('tau\\[',   nam)]
      alphas <- parm[al]
      betas  <- parm[be]
      betas  <- matrix(betas, nrow=1) %*% t(wqrX$Rinv)
      names(betas)  <- atr$colnames
      alphas <- alphas - sum(betas * xbar)
      if(pppo) {
        if(length(cppo)) { # constrained PPO model
          taus <- matrix(parm[ta], nrow=pppo, ncol=1)
          taus <- wqrZ$Rinv %*% taus
          alphas <- alphas - d$pposcore[-1] * sum(taus * zbar)
          # zbartau <- sum(taus * zbar)    # longhand
          # for(j in 1 : (k-1)) alphas[j] <- alphas[j] - d$pposcore[j + 1] * zbartau
          subscript <- as.integer(gsub('tau\\[(.*)\\]', '\\1', ta))  # Z column
          namtau <- paste(colnames(Z)[subscript], 'x f(y)')
        }
        else {
          taus  <- matrix(parm[ta], nrow=pppo, ncol=k-2)
          taus  <- wqrZ$Rinv %*% taus
          alphas[-1] <- alphas[-1] - matrix(zbar, ncol=pppo) %*% taus
          ro     <- as.integer(gsub('tau\\[(.*),.*',    '\\1', ta))  # y cutoff
          co     <- as.integer(gsub('tau\\[.*,(.*)\\]', '\\1', ta))  # Z column
          namtau <- paste0(colnames(Z)[ro], ':y>=', ylev[-(1:2)][co])
          }
        names(taus) <- namtau
      }
      names(alphas) <- if(nrp == 1) 'Intercept' else paste0('y>=', ylev[-1])
      opt <- list(coefficients=c(alphas, betas, taus), cppo=cppo, zbar=zbar,
                  sigmag=parm[sigmagname], deviance=-2 * g$value,
                  return_code=g$return_code, hessian=g$hessian)
      }
    if(method == 'optimizing') return(opt)
  }

  if(progress != '') sink(progress, append=TRUE)

  debug(1)
  exclude <- c('sigmaw', 'gamma_raw', 'pi')
  if(! loo) exclude <- c(exclude, 'log_lik')
  g <- rstan::sampling(mod, pars=exclude, include=FALSE,
                       data=d, iter=iter, chains=chains, refresh=refresh,
                       init=inits, ...)

  debug(2)

  if(progress != '') sink()
	nam <- names(g)
	al  <- nam[grep('alpha\\[', nam)]
	be  <- nam[grep('beta\\[', nam)]
  ga  <- nam[grep('gamma\\[', nam)]
  draws  <- as.matrix(g)
  ndraws <- nrow(draws)
	alphas <- draws[, al, drop=FALSE]
	betas  <- draws[, be, drop=FALSE]
  betas  <- betas %*% t(wqrX$Rinv)

  omega   <- NULL      # non-intercepts, non-slopes
  clparm  <- character(0)
  gammas  <- NULL

  if(length(cluster)) {
    omega   <- cbind(sigmag = draws[, sigmagname])
    clparm  <- 'sigmag'
    cle     <- draws[, ga, drop=FALSE]
    gammas  <- apply(cle, 2, median)      # posterior median per subject
  }

  debug(3)
  tau <- ta <- tauInfo <- NULL
  if(pppo) {
    ta     <- nam[grep('tau\\[', nam)]
    clparm <- c(clparm, ta)
    if(length(cppo)) {  # constrained partial PO model
      subscript <- as.integer(gsub('tau\\[(.*)\\]', '\\1', ta))  # Z column
      xt     <- colnames(Z)[subscript]
      namtau <- paste(xt, 'x f(y)')
      taus   <- draws[, ta, drop=FALSE]
      taus   <- taus %*% t(wqrZ$Rinv)
      tauInfo <- data.frame(name=namtau, x=xt)
      ## Make sure mtaus is not referenced later when length(cppo) > 0
    }
    else {
      ro     <- as.integer(gsub('tau\\[(.*),.*',    '\\1', ta))  # y cutoff
      co     <- as.integer(gsub('tau\\[.*,(.*)\\]', '\\1', ta))  # Z column
      xt     <- colnames(Z)[ro]
      yt     <- ylev[-(1:2)][co]
      namtau <- paste0(xt, ':y>=', yt)
      taus   <- draws[, ta, drop=FALSE]
      tauInfo <- data.frame(intercept=1 + co, name=namtau, x=xt, y=yt)
      ## Need taus as a 3-dim array for intercept correction for centering
      ## Also helps with undoing the QR orthogonalization
      mtaus      <- array(taus, dim=c(ndraws, pppo, k - 2))

      for(j in 1 : (k - 2))
        mtaus[, , j] <- dropadim(mtaus, 3, j) %*% t(wqrZ$Rinv)
      taus <- matrix(as.vector(mtaus), nrow=ndraws, ncol=pppo * (k -2))
      }
    colnames(taus) <- namtau
    }

  debug(4)

  epsmed <- NULL

  debug(5)
  diagnostics <-
		tryCatch(rstan::summary(g, pars=c(al, be, clparm), probs=NULL),
             error=function(...) list(fail=TRUE))
  if(itfailed(diagnostics)) {
    warning('rstan::summary failed; see fit component diagnostics')
    diagnostics <- list(pars=c(al, be, clparm), failed=TRUE)
  } else diagnostics <- diagnostics$summary[,c('n_eff', 'Rhat')]

  debug(6)
  if(length(ppairs)) {
    if(is.character(ppairs)) png(ppairs, width=1000, height=1000, pointsize=11)
    pairs(g, pars=c(al, be, clparm))
    if(is.character(ppairs)) dev.off()
    }

  debug(7)
	# Back-scale to original data scale
	alphacorr <- rowSums(sweep(betas, 2, xbar, '*')) # was xbar/xsd
	alphas    <- sweep(alphas, 1, alphacorr, '-')
  if(length(cppo)) {
    sc <- d$pposcore[-1]
    for(i in 1 : ndraws) alphas[i, ] <- alphas[i,,drop=FALSE] - sc * sum(taus[i, ] * zbar)
    }
  else if(pppo > 0)
    for(i in 1 : ndraws)
      for(j in 3 : k)
        alphas[i, j - 1] <- alphas[i, j - 1, drop=FALSE] -
                            sum(mtaus[i, , j-2] * zbar)

  #  alphas[, -1] <- alphas[, -1, drop=FALSE] - zalphacorr
	colnames(alphas) <- if(nrp == 1) 'Intercept' else paste0('y>=', ylev[-1])
	colnames(betas)  <- atr$colnames

	draws            <- cbind(alphas, betas, taus)

  debug(8)
  param <- rbind(mean=colMeans(draws), median=apply(draws, 2, median))
  if(method != 'sampling') {
    param <- rbind(mode=opt$coefficients, param)
    opt$coefficients <- NULL
  }

  Loo <- lootime <- NULL
  if(loo) {
    lootime <- system.time(
      Loo <- tryCatch(rstan::loo(g), error=function(...) list(fail=TRUE)))
    if(itfailed(Loo)) {
      warning('loo failed; try running on loo(stanGet(fit object)) for more information')
      Loo <- NULL
      }
  }

  debug(9)

	res <- list(call=call, fitter=fitter, link='logistic',
							draws=draws, omega=omega,
              gammas=gammas, eps=epsmed,
              param=param, priorsd=priorsd, priorsdppo=priorsdppo,
              psigma=psigma, rsdmean=rsdmean, rsdsd=rsdsd, conc=conc,
              notransX=notransXn, notransZ=notransZn, xqrsd=xqrsd,
              N=n, Ncens=Ncens, p=p, pppo=pppo, cppo=cppo, yname=yname,
              ylevels=ylev, freq=numy,
						  alphas=al, betas=be, taus=ta, tauInfo=tauInfo,
						  xbar=xbar, zbar=zbar, Design=atr, zDesign=zatr,
              scale.pred=c('log odds', 'Odds Ratio'),
							terms=Terms, assign=ass, zassign=zass,
              na.action=atrx$na.action, fail=FALSE,
							non.slopes=nrp, interceptRef=kmid,
              sformula=sformula, zsformula=zsformula,
							x=if(x) X, y=if(y) Y, z=if(x) Z,
              loo=Loo, lootime=lootime,
              clusterInfo=if(length(cluster))
                list(cluster=if(x) cluster else NULL, n=Nc, name=clustername),
							opt=opt, diagnostics=diagnostics,
              iter=iter, chains=chains, stancode=stancode, datahash=datahash)
	class(res) <- c('blrm', 'rmsb', 'rms')
  if(length(file)) saveRDS(res, file, compress='xz')
  res$rstan <- g
	res
}

##' Compute Indexes of Predictive Accuracy and Their Uncertainties
##'
##' For a binary or ordinal logistic regression fit from [blrm()], computes several indexes of predictive accuracy along with highest posterior density intervals for them.  Optionally plots their posterior densities.
##' When there are more than two levels of the outcome variable, computes Somers' Dxy and c-index on a random sample of 10,000 observations.
##' @title blrmStats
##' @param fit an object produced by [blrm()]
##' @param ns number of posterior draws to use in the calculations (default is 400)
##' @param prob HPD interval probability (default is 0.95)
##' @param pl set to `TRUE` to plot the posterior densities using base graphics
##' @param dist if `pl` is `TRUE` specifies whether to plot the density estimate (the default) or a histogram
##' @return list of class `blrmStats` whose most important element is `Stats`.  The indexes computed are defined below, with gp, B, EV, and vp computed using the intercept corresponding to the median value of Y.  See <https://fharrell.com/post/addvalue> for more information.
##' \describe{
##'  \item{"Dxy"}{Somers' Dxy rank correlation between predicted and observed.  The concordance probability (c-index; AUROC in the binary Y case) may be obtained from the relationship Dxy=2(c-0.5).}
##'  \item{"g"}{Gini's mean difference: the average absolute difference over all pairs of linear predictor values}
##'  \item{"gp"}{Gini's mean difference on the predicted probability scale}
##'  \item{"B"}{Brier score}
##'  \item{"EV"}{explained variation}
##'  \item{"v"}{variance of linear predictor}
##'  \item{"vp"}{variable of estimated probabilities}
##' }
##' @seealso [Hmisc::rcorr.cens()]
##' @examples
##' \dontrun{
##'   f <- blrm(...)
##'   blrmStats(f, pl=TRUE)   # print and plot
##' }
##' @author Frank Harrell
##' @export
blrmStats <- function(fit, ns=400, prob=0.95, pl=FALSE,
                      dist=c('density', 'hist')) {
  dist <- match.arg(dist)

  f <- fit[c('x', 'y', 'z', 'non.slopes', 'interceptRef', 'pppo', 'cppo',
             'draws', 'ylevels', 'tauInfo')]
  X <- f$x
  Z <- f$z
  y <- f$y
  if(length(X) == 0 | length(y) == 0)
    stop('must have specified x=TRUE, y=TRUE to blrm')
  if(is.matrix(y)) {
    if(any(y[, 1] != y[, 2])) warning('some observations are censored but only left endpoint used in summary statistics')
    y <- y[, 1]
    }
  y <- as.integer(y) - 1
  nrp    <- f$non.slopes
  kmid   <- f$interceptRef  # intercept close to median
  s      <- tauFetch(f, intercept=kmid, what='nontau')
  # old fit objects did not include pppo for non-PPO models:
  pppo   <- if(! length(f$pppo)) 0 else f$pppo
  ## tauFetch automatically multiplies tau by cppo function if present
  if(pppo > 0) stau <- tauFetch(f, intercept=kmid, what='tau')
  ndraws <- nrow(s)
  ns     <- min(ndraws, ns)
  if(ns < ndraws) {
    j <- sample(1 : ndraws, ns, replace=FALSE)
    s <- s[j,, drop=FALSE]
    if(pppo > 0) stau <- stau[j,, drop=FALSE]
    }
  ylev  <- f$ylevels
  ybin  <- length(ylev) == 2
  stats <- matrix(NA, nrow=ns, ncol=8)
  colnames(stats) <- c('Dxy', 'C', 'g', 'gp', 'B', 'EV', 'v', 'vp')
  dxy <- if(length(ylev) == 2)
           function(x, y) somers2(x, y)['Dxy']
         else
           function(x, y) {
             con <- survival::survConcordance.fit(survival::Surv(y), x)
             conc <- con['concordant']; disc <- con['discordant']
             - (conc - disc) / (conc + disc)
             }
  brier <- function(x, y) mean((x - y) ^ 2)
  br2   <- function(p) var(p) / (var(p) + sum(p * (1 - p)) / length(p))

  nobs <- length(y)
  is <- if((nobs <= 10000) || (length(ylev) == 2)) 1 : nobs else
              sample(1 : nobs, 10000, replace=FALSE)

  for(i in 1 : ns) {
    beta  <- s[i,, drop=FALSE ]
    lp    <- cbind(1, X) %*% t(beta)
    if(pppo > 0) {
      tau <- stau[i,, drop=FALSE]
      lp  <- lp + Z %*% t(tau)
      }
    prb  <- plogis(lp)
    d    <- dxy(lp[is], y[is])
    C    <- (d + 1.) / 2.
    st <- c(d, C, GiniMd(lp), GiniMd(prb),
            brier(prb, y > kmid - 1),
            br2(prb), var(lp), var(prb))
    stats[i,] <- st
  }
  sbar  <- colMeans(stats)
  se    <- apply(stats, 2, sd)
  hpd   <- apply(stats, 2, HPDint, prob=prob)
  sym   <- apply(stats, 2, distSym)

  text <- paste0(round(sbar, 3), ' [', round(hpd[1,], 3), ', ',
                 round(hpd[2,], 3), ']')
  Stats <- rbind(Mean=sbar, SE=se, hpd=hpd, Symmetry=sym)
  statnames <- colnames(Stats)
  names(text) <- statnames

  if(pl) {
    oldpar <- par(no.readonly=TRUE)
    on.exit(par(oldpar))
    par(mfrow=c(4,2), mar=c(3, 2, 0.5, 0.5), mgp=c(1.75, .55, 0))
    for(w in setdiff(statnames, 'C')) {
      p <- switch(dist,
             density = {
               den <- density(stats[, w])
               plot(den, xlab=w, ylab='', type='l', main='') },
             hist = {
               den <- hist(stats[, w], probability=TRUE,
                           nclass=100, xlab=w, ylab='', main='')
               den$x <- den$breaks; den$y <- den$density } )
      ref <- c(sbar[w], hpd[1, w], hpd[2, w])
      abline(v=ref, col=gray(0.85))
      text(min(den$x), max(den$y), paste0('Symmetry:', round(sym[w], 2)),
           adj=c(0, 1), cex=0.7)
    }
  }
  structure(list(stats=Stats, text=text, ndraws=ndraws, ns=ns,
                 non.slopes=nrp, intercept=kmid), class='blrmStats')
  }

##' Print Details for `blrmStats` Predictive Accuracy Measures
##'
##' Prints results of `blrmStats` with brief explanations
##' @title print.blrmStats
##' @param x an object produced by `blrmStats`
##' @param dec number of digits to round indexes
##' @param ... ignored
##' @examples
##' \dontrun{
##'   f <- blrm(...)
##'   s <- blrmStats(...)
##'   s    # print with defaults
##'   print(s, dec=4)
##' }
##' @author Frank Harrell
##' @export
print.blrmStats <- function(x, dec=3, ...) {
  ns <- x$ns; ndraws <- x$ndraws
  if(ns < ndraws)
    cat('Indexes computed for a random sample of', ns, 'of', ndraws,
        'posterior draws\n\n')
  else
    cat('Indexes computed on', ndraws, 'posterior draws\n\n')

  if(x$non.slopes > 1)
    cat('gp, B, EV, and vp are for intercept', x$intercept,
        'out of', x$non.slopes, 'intercepts\n\n')
  print(round(x$stats, dec))
  cat('\nDxy: 2*(C - 0.5)   C: concordance probability',
      'g: Gini mean |difference| on linear predictor (lp)',
      'gp: Gini on predicted probability        B: Brier score',
      'EV: explained variation on prob. scale   v: var(lp)   vp: var(prob)\n',
      sep='\n')
    }


##' Print [blrm()] Results
##'
##' Prints main results from [blrm()] along with indexes and predictive accuracy and their highest posterior density intervals computed from `blrmStats`.
##' @title print.blrm
##' @param x object created by [blrm()]
##' @param dec number of digits to print to the right of the decimal
##' @param coefs specify `FALSE` to suppress printing parameter estimates, and in integer k to print only the first k
##' @param intercepts set to `FALSE` to suppress printing intercepts.   Default is to print them unless there are more than 9.
##' @param prob HPD interval probability for summary indexes
##' @param ns number of random samples of the posterior draws for use in computing HPD intervals for accuracy indexes
##' @param title title of output, constructed by default
##' @param ... passed to `prModFit`
##' @examples
##' \dontrun{
##'   f <- blrm(...)
##'   options(lang='html')   # default is lang='plain'; also can be latex
##'   f               # print using defaults
##'   print(f, posterior.summary='median')   # instead of post. means
##' }
##' @author Frank Harrell
##' @export
print.blrm <- function(x, dec=4, coefs=TRUE, intercepts=x$non.slopes < 10,
                       prob=0.95, ns=400, title=NULL, ...) {
  latex <- prType() == 'latex'

  if(! length(title))
    title <- paste(if(length(x$cppo)) 'Constrained',
                   if(x$pppo > 0)     'Partial',
                   if(length(x$ylevels) > 2) 'Proportional Odds Ordinal',
                                      'Logistic Model')
  z <- list()
  k <- 0

  if(length(x$freq) > 3 && length(x$freq) < 50) {
    k <- k + 1
    z[[k]] <- list(type='print', list(x$freq),
                   title='Frequencies of Responses')
  }
  if(length(x$na.action)) {
    k <- k + 1
    z[[k]] <- list(type=paste('naprint',class(x$na.action),sep='.'),
                   list(x$na.action))
  }
  qro <- function(x) {
    r <- round(c(median(x), HPDint(x, prob)), 4)
    paste0(r[1], ' [', r[2], ', ', r[3], ']')
    }
  ci  <- x$clusterInfo
  sigmasum <- NULL
  if(length(ci)) sigmasum <- qro(x$omega[, 'sigmag'])

  loo <- x$loo
 elpd_loo <- p_loo <- looic <- NULL
  if(length(loo)) {
    lo <- loo$estimates
    pm <- if(prType() == 'plain') '+/-' else
              markupSpecs[[prType()]][['plminus']]
    nlo <- rownames(lo)
     lo <- paste0(round(lo[, 'Estimate'], 2), pm, round(lo[, 'SE'], 2))
    elpd_loo <- lo[1]; p_loo <- lo[2]; looic <- lo[3]
  }
  Ncens <- x$Ncens
  L     <- if(Ncens[1] > 0) paste0('L=', Ncens[1])
  R     <- if(Ncens[2] > 0) paste0('R=', Ncens[2])
  int   <- if(Ncens[3] > 0) paste0('I=', Ncens[3])
  ce    <- if(sum(Ncens) > 0) sum(Ncens)
  ced   <- if(sum(Ncens) > 0)
             paste0(paste(c(L, R, int), collapse=', '))
  misc <- rms::reListclean(Obs             = x$N,
                      Censored        = ce,
                      ' '             = ced,
                      Draws           = nrow(x$draws),
                      Chains          = x$chains,
                      Imputations     = x$n.impute,
                      p               = x$p,
                      'Cluster on'    = ci$name,
                      Clusters        = ci$n,
                      'sigma gamma'   = sigmasum)

  if(length(x$freq) < 4) {
    names(x$freq) <- paste(if(latex)'~~' else ' ',
                           names(x$freq), sep='')
    misc <- c(misc[1], x$freq, misc[-1])
  }
  a <- blrmStats(x, ns=ns)$text
  mixed <- rms::reListclean('LOO log L'  = elpd_loo,
                      'LOO IC'      = looic,
                      'Effective p' = p_loo,
                      B             = a['B'])

  disc <- rms::reListclean(g       = a['g'],
                      gp      = a['gp'],
                      EV      = a['EV'],
                      v       = a['v'],
                      vp      = a['vp'])

  discr <- rms::reListclean(C       = a['C'],
                      Dxy     = a['Dxy'])


  headings <- c('','Mixed Calibration/\nDiscrimination Indexes',
                   'Discrimination\nIndexes',
                   'Rank Discrim.\nIndexes')

  data <- list(misc, c(mixed, NA), c(disc, NA), c(discr, NA))
  k <- k + 1
  z[[k]] <- list(type='stats', list(headings=headings, data=data))

  if(coefs) {
    k <- k + 1
    z[[k]] <- list(type='coefmatrix',
                   list(bayes=print.rmsb(x, prob=prob, intercepts=intercepts,
                                         pr=FALSE)))
  }

  footer <- if(length(x$notransX))
              paste('The following parameters remained separate (where not orthogonalized) during model fitting so that prior distributions could be focused explicitly on them:',
                    paste(x$notransX, collapse=', '))
  rms::prModFit(x, title=title, z, digits=dec, coefs=coefs, footer=footer, ...)
}

##' Make predictions from a [blrm()] fit
##'
##' Predict method for [blrm()] objects
##' @title predict.blrm
##' @param object,...,type,se.fit,codes see [predict.lrm]
##' @param posterior.summary set to `'median'` or `'mode'` to use posterior median/mode instead of mean
##' @param cint probability for highest posterior density interval
##' @return a data frame,  matrix, or vector with posterior summaries for the requested quantity, plus an attribute `'draws'` that has all the posterior draws for that quantity.  For `type='fitted'` and `type='fitted.ind'` this attribute is a 3-dimensional array representing draws x observations generating predictions x levels of Y.
##' @examples
##' \dontrun{
##'   f <- blrm(...)
##'   predict(f, newdata, type='...', posterior.summary='median')
##' }
##' @seealso [predict.lrm]
##' @author Frank Harrell
##' @export
predict.blrm <- function(object, ...,
		type=c("lp","fitted","fitted.ind","mean","x","data.frame",
		  "terms", "cterms", "ccterms", "adjto", "adjto.data.frame",
      "model.frame"),
		se.fit=FALSE, codes=FALSE,
    posterior.summary=c('mean', 'median'), cint=0.95) {

  type              <- match.arg(type)
  posterior.summary <- match.arg(posterior.summary)

  if(se.fit) warning('se.fit does not apply to Bayesian models')

  pppo <- object$pppo
#  if(pppo > 0 && (type %in% c('lp', 'mean', 'fitted', 'fitted.ind')))
#    stop('type not yet implemented for partial proportional odds models')

  if(type %in% c('lp', 'x', 'data.frame', 'terms', 'cterms', 'ccterms',
                 'adjto', 'adjto.data.frame', 'model.frame'))
    return(rms::predictrms(object,...,type=type,
                      posterior.summary=posterior.summary))

  if(type %nin% c('mean', 'fitted', 'fitted.ind'))
    stop('types other than mean, fitted, fitted.ind not yet implemented')

  ns      <- object$non.slopes
  tauinfo <- object$tauInfo
  cppo    <- object$cppo

  if(ns == 1 & type == "mean")
    stop('type="mean" makes no sense with a binary response')

  draws   <- object$draws
  ndraws  <- nrow(draws)
  cn      <- colnames(draws)
  ints    <- draws[, 1 : ns, drop=FALSE]
  betanam <- setdiff(cn, c(cn[1:ns], tauinfo$name))
  betas   <- draws[, betanam, drop=FALSE]
  if(pppo > 0) {
    taunam <- tauinfo$name
    taus   <- draws[, taunam, drop=FALSE]
    }

  X <- rms::predictrms(object, ..., type='x',
                       posterior.summary=posterior.summary)
  if(pppo > 0) {
    Z <- X$z
    X <- X$x
    }
  rnam  <- rownames(X)
  n     <- nrow(X)

  ## Get cumulative probability function used
  link  <- object$link
  cumprob <- rms::probabilityFamilies[[link]]$cumprob

  if(ns == 1) return(cumprob(ints + betas %*% t(X)))  # binary logistic model

  cnam  <- cn[1:ns]
  ylev  <- object$ylevels
  # First intercept corresponds to second distinct Y value
  if(length(cppo)) cppos <- cppo(ylev[-1])

  if(type == 'mean') {
    if(codes) vals <- 1:length(ylev)
    else {
      vals <- as.numeric(ylev)
      if(any(is.na(vals)))
         stop('values of response levels must be numeric for type="mean" and codes=FALSE')
    }
  }

  ynam <- paste(object$yname, "=", ylev, sep="")
  PP   <- array(NA, dim=c(ndraws, n, ns),
                dimnames=list(NULL, rnam, cnam))
  PPeq <- array(NA, dim=c(ndraws, n, ns + 1),
                dimnames=list(NULL, rnam, ynam))

  M  <- matrix(NA, nrow=ndraws, ncol=n, dimnames=list(NULL, rnam))

  for(i in 1 : ndraws) {
    inti    <- ints[i,]               # alphas for ith draw
    betai   <- betas[i,,drop=FALSE]
    if(pppo > 0) {
      taui <- taus[i,, drop=FALSE]
      zt   <- taui %*% t(Z)           # 1xq qxn = 1xn
      ## Convenient to think of deviations from proportional odds as
      ## changes in intercepts
      inti <- inti + sapply(cppos, '*', zt)
      }
    xb      <- betai %*% t(X)         # 1xn

    ## sapply(U (u-vector), '+', V (v-vector)) = v x u matrix
    ## column j = all V with U[j] added
    xb      <- sapply(inti, "+", xb)  # n x ns matrix, col j adds intercept j

    P       <- cumprob(xb)            # preserves matrix;  1 x nrow(X)
    PP[i,,] <- P
    if(type != 'fitted') {
      Peq       <- cbind(1, P) - cbind(P, 0)
      PPeq[i,,] <- Peq
      if(type == 'mean') M[i, ] <- drop(Peq %*% vals)
      }
  }

  ps <- switch(posterior.summary, mean=mean, median=median)
  h <- function(x) {
    s  <- ps(x)
    ci <- HPDint(x, cint)
    r  <- c(s, ci)
    names(r)[1] <- posterior.summary
    r
    }

  ## Function to summarize a 3-d array and transform the results
  ## to a data frame with variables x (row names of predictions), y level,
  ## mean/median, Lower, Upper
  ## Input is ndraws x # predicton requests x # y categories (less 1 if cum.p.)

  s3 <- function(x) {
    yl <- if(type == 'fitted') cnam else ynam
    d <- expand.grid(y = yl, x = rnam, stat=NA, Lower=NA, Upper=NA,
                     stringsAsFactors=FALSE)
    for(i in 1 : nrow(d)) {
      u <- h(x[, d$x[i], d$y[i]])
      d$stat[i] <- u[1]
      d$Lower[i] <- u['Lower']
      d$Upper[i] <- u['Upper']
    }
    names(d)[3] <- upFirst(posterior.summary)
    d
  }

  ## Similar for a 2-d array
  s2 <- function(x) {
    d <- expand.grid(x = rnam, stat=NA, Lower=NA, Upper=NA,
                     stringsAsFactors=FALSE)
    for(i in 1 : nrow(d)) {
      u <- h(x[, d$x[i]])
      d$stat[i]  <- u[1]
      d$Lower[i] <- u['Lower']
      d$Upper[i] <- u['Upper']
    }
    names(d)[2] <- Hmisc::upFirst(posterior.summary)
    d
  }


  r <- switch(type,
              fitted     =  structure(s3(PP), draws=PP),
              fitted.ind =  structure(s3(PPeq), draws=PPeq),
              mean       =  structure(s2(M), draws=M))

  class(r) <- c('predict.blrm', class(r))
  r
}

##' Print Predictions for [blrm()]
##'
##' Prints the summary portion of the results of `predict.blrm`
##' @title print.predict.blrm
##' @param x result from `predict.blrm`
##' @param digits number of digits to round numeric results
##' @param ... ignored
##' @author Frank Harrell
##' @export
print.predict.blrm <- function(x, digits=3, ...) {
  numvar <- sapply(x, is.numeric)
  numvar <- names(numvar)[numvar]
  for(j in numvar) x[, j] <- round(x[, j], digits)
  print.data.frame(x)
}


##' Function Generator for Mean Y for [blrm()]
##'
##' Creates a function to turn a posterior summarized linear predictor lp (e.g. using posterior mean of intercepts and slopes) computed at the reference intercept into e.g. an estimate of mean Y using the posterior mean of all the intercept.  `lptau` must be provided when call the created function if the model is a partial proportional odds model.
##' @title Mean.blrm
##' @param object a [blrm()] fit
##' @param codes if `TRUE`, use the integer codes \eqn{1,2,\ldots,k} for the \eqn{k}-level response in computing the predicted mean response.
##' @param posterior.summary defaults to posterior mean; may also specify `"median"`.  Must be consistent with the summary used when creating `lp`.
##' @param ... ignored
##' @return an R function
##' @author Frank Harrell
##' @method Mean blrm
##' @export
Mean.blrm <- function(object, codes=FALSE,
                      posterior.summary=c('mean', 'median'), ...) {
  posterior.summary <- match.arg(posterior.summary)
  ns <- Hmisc::num.intercepts(object)
  if(ns < 2)
    stop('using this function only makes sense for >2 ordered response categories')

  ## Get cumulative probability function used
  link    <- object$link
  cumprob <- probabilityFamilies[[link]]$cumprob

  cof <- coef(object, stat=posterior.summary)
  intercepts <- cof[1 : ns]

  f <- function(lp=numeric(0), lptau=numeric(0),
                intercepts=numeric(0), ylevels=numeric(0), codes=FALSE,
                interceptRef=integer(0), cppo=NULL, cumprob=cumprob) {
    ns <- length(intercepts)
    ## assume lp is computed at the reference intercept
    ## subtract the reference intercept so can use all intercepts
    lp <- lp - intercepts[interceptRef]
    xb <- sapply(intercepts, '+', lp)   # n x ns matrix; adds all ints to lp

    if(length(lptau)) {
      cppos <- cppo(ylevels[-1])   # first intercept corresponds to 2nd Y
      for(j in 1 : ns)
        xb[, j] <- xb[, j] + cppos[j] * lptau[j]
      }

    if(codes) ylevels <- 1 : length(ylevels)
    else {
      ylevels <- as.numeric(ylevels)
      if(any(is.na(ylevels)))
        stop('values of response levels must be numeric when codes=FALSE')
    }

    P  <- matrix(cumprob(xb), ncol=ns)
    P  <- cbind(1, P) - cbind(P, 0)
    m  <- drop(P %*% ylevels)
    names(m) <- names(lp)
    m
  }

  ir <- object$interceptRef
  if(! length(ir)) ir <- 1
  formals(f) <- list(lp=numeric(0), lptau=numeric(0), intercepts=intercepts,
                     ylevels=object$ylevels, codes=codes,
                     interceptRef=ir, cppo=object$cppo,
                     cumprob=cumprob)
  f
}

##' Function Generator for Quantiles of Y for [blrm()]
##'
##' Creates a function to turn a posterior summarized linear predictor lp (e.g. using posterior mean of intercepts and slopes) computed at the reference intercept into e.g. an estimate of a quantile of Y using the posterior mean of all the intercepts.  `lptau` must be provided when call the created function if the model is a partial proportional odds model.
##' @title Quantile.blrm
##' @param object a [blrm()] fit
##' @param codes if `TRUE`, use the integer codes \eqn{1,2,\ldots,k} for the \eqn{k}-level response in computing the quantile
##' @param posterior.summary defaults to posterior mean; may also specify `"median"`.  Must be consistent with the summary used when creating `lp`.
##' @param ... ignored
##' @return an R function
##' @author Frank Harrell
##' @method Quantile blrm
##' @export
Quantile.blrm <- function(object, codes=FALSE,
                      posterior.summary=c('mean', 'median'), ...) {
  posterior.summary <- match.arg(posterior.summary)
  ns <- Hmisc::num.intercepts(object)
  if(ns < 2)
    stop('using this function only makes sense for >2 ordered response categories')

  ## Get cumulative probability function used
  link    <- object$link
  pfam    <- rms::probabilityFamilies[[link]]
  cumprob <- pfam$cumprob
  inverse <- pfam$inverse

  cof        <- coef(object, stat=posterior.summary)
  intercepts <- cof[1 : ns]

  f <- function(q=.5, lp=numeric(0), lptau=numeric(0),
                intercepts=numeric(0), ylevels=numeric(0), codes=FALSE,
                interceptRef=integer(0), cppo=NULL,
                cumprob=NULL, inverse=NULL) {

    if(codes) ylevels <- 1 : length(ylevels)
    else {
      ylevels <- as.numeric(ylevels)
      if(any(is.na(ylevels)))
        stop('values of response levels must be numeric when codes=FALSE')
    }

    ## Solve inverse(1 - q) = a + xb; inverse(1 - q) - xb = a
    ## Shift intercepts to left one position because quantile
    ## is such that Prob(Y <= y) = q whereas model is stated in
    ## Prob(Y >= y)
    lp <- lp - intercepts[interceptRef]

    if(length(lptau)) {
      ## Don't know how to handle this over all observations
      ## Do for one observation at a time
      cppos <- cppo(ylevels[-1])
      n     <- length(lptau)
      quant <- numeric(n)
      for(i in 1 : length(lptau)) {
        ints <- intercepts + cppos * lptau[i]
        quant[i] <- approx(c(cumprob(ints + lp[i]), 0), ylevels,
                           xout=1 - q, rule=2)$y
        }
      }
    else {
        ## Interpolation on linear predictors scale:
        ## z <- approx(c(intercepts, -1e100), values,
        ##             xout=inverse(1 - q) - lp, rule=2)$y
        ## Interpolation approximately on Y scale:
        lpm <- mean(lp, na.rm=TRUE)
        quant <- approx(c(cumprob(intercepts + lpm), 0), ylevels,
                        xout=cumprob(inverse(1 - q) - lp + lpm), rule=2)$y
    }
    names(quant) <- names(lp)
    quant
  }
  trans <- object$trans
  formals(f) <- list(q=.5, lp=numeric(0), lptau=numeric(0),
                     intercepts=intercepts,
                     ylevels=object$ylevels, codes=codes,
                     interceptRef=object$interceptRef,
                     cppo=object$cppo, cumprob=cumprob, inverse=inverse)
  f
}

##' Function Generator for Exceedance Probabilities for [blrm()]
##'
##' For a [blrm()] object generates a function for computing the	estimates of the function Prob(Y>=y) given one or more values of the linear predictor using the reference (median) intercept.  This function can optionally be evaluated at only a set of user-specified	`y` values, otherwise a right-step function is returned.  There is a plot method for plotting the step functions, and if more than one linear predictor was evaluated multiple step functions are drawn. `ExProb` is especially useful for `nomogram()`.  The linear predictor argument is a posterior summarized linear predictor lp (e.g. using posterior mean of intercepts and slopes) computed at the reference intercept.  `lptau` must be provided when call the created function if the model is a partial proportional odds model.
##' @title ExProb.blrm
##' @param object a [blrm()] fit
##' @param posterior.summary defaults to posterior mean; may also specify `"median"`.  Must be consistent with the summary used when creating `lp`.
##' @param ... ignored
##' @return an R function
##' @author Frank Harrell
##' @method ExProb blrm
##' @export
ExProb.blrm <- function(object,
                        posterior.summary=c('mean', 'median'), ...) {
  posterior.summary <- match.arg(posterior.summary)
  ns <- num.intercepts(object)
  if(ns < 2)
    stop('using this function only makes sense for >2 ordered response categories')

  ## Get cumulative probability function used
  link    <- object$link
  pfam    <- rms::probabilityFamilies[[link]]
  cumprob <- pfam$cumprob

  ylevels    <- object$ylevels
  cof        <- coef(object, stat=posterior.summary)
  intercepts <- cof[1 : ns]

  f <- function(lp=numeric(0), lptau=numeric(0),
                y=NULL, intercepts=numeric(0),
                ylevels=numeric(0),
                interceptRef=integer(0), cppo=NULL,
                cumprob=NULL, yname=NULL) {
    lp <- lp - intercepts[interceptRef]

    n <- length(lp)
    yeval <- if(length(y)) y else ylevels
    prob <- matrix(NA, nrow=n, ncol=length(yeval))
    if(length(yeval) > 1) colnames(prob) <- paste0('Prob(Y>=', yeval, ')')
    yisnum <- all.is.numeric(ylevels)
    if(length(lptau)) cppos <- cppo(ylevels[-1])

      for(i in 1 : n) {
        xb <- intercepts + lp[i]
        if(length(lptau)) xb <- xb + cppos * lptau[i]
        cp <- c(1., cumprob(xb))  # was if(! length(y))...
        if(yisnum) {
          cp[yeval <= min(ylevels)] <- 1.
          cp[yeval >  max(ylevels)] <- 0.
        }
        if(length(y)) {
          cp <- if(yisnum)
                  approx(ylevels, cp, xout=yeval, f=1, method='constant')$y
          else cp[match(yeval, ylevels)]
          }
        prob[i, ] <- cp
      }
      res <- if(length(yeval) == 1) drop(prob) else
                    list(y=ylevels, prob=prob, yname=yname)
      structure(res, class='ExProb')
  }
  formals(f) <- list(lp=numeric(0), lptau=numeric(0),
                     y=NULL, intercepts=intercepts,
                     ylevels=ylevels,
                     interceptRef=object$interceptRef,
                     cppo=object$cppo, cumprob=cumprob, yname=object$yname)
  f
}

##' Fetch Partial Proportional Odds Parameters
##'
##' Fetches matrix of posterior draws for partial proportional odds parameters (taus) for a given intercept.  Can also form a matrix containing both regular parameters and taus, or for just non-taus.  For the constrained partial proportional odds model the function returns the appropriate `cppo` function value multiplied by tau (tau being a vector in this case and not a matrix).
##' @title tauFetch
##' @param fit an object created by [blrm()]
##' @param intercept integer specifying which intercept to fetch
##' @param what specifies the result to return
##' @return matrix with number of raws equal to the numnber of original draws
##' @author Frank Harrell
##' @export
tauFetch <- function(fit, intercept, what=c('tau', 'nontau', 'both')) {
  what   <- match.arg(what)
  f      <- fit[c('tauInfo', 'draws', 'non.slopes', 'cppo', 'ylevels')]
  info   <- f$tauInfo
  draws  <- f$draws
  nd     <- nrow(draws)
  cn     <- colnames(draws)
  int    <- intercept
  nints  <- f$non.slopes
  if(int > nints) stop('intercept is too large')
  ## Keep only the intercept of interest
  cn     <- c(cn[int], cn[-(1 : nints)])
  nontau <- setdiff(cn, info$name)
  cppo   <- f$cppo
  if(length(cppo)) {   # constrained PPO model
    taunames <- info$name
    if(what != 'nontau') {
      ## First intercept is for 2nd ordered distinct Y value
      ## cppo argument is on original y scale
      cppoint  <- cppo(f$ylevels[int + 1])
      draws[, taunames] <- cppoint * draws[, taunames, drop=FALSE]
      }
    draws[, switch(what,
                   tau=taunames, nontau=nontau, both=c(nontau, taunames)),
          drop=FALSE]
  }
  else {
    taunames   <- if(int > 1) subset(info, intercept == int)$name
    ## Partial proportional odds parameters start with the 2nd intercept
    ## for the unconstrained PPO model
    ## taunames is absent if intercept=1 and unconstrained PPO
    ## Use # taus for intercept=2 in this case
    switch(what,
           tau    = if(! length(taunames))
                      matrix(0, nrow=nd, ncol=sum(info$intercept == 2))
                    else draws[, taunames, drop=FALSE],
           nontau = draws[, nontau, drop=FALSE],
           both   = if(! length(taunames))
                      cbind(draws[, nontau, drop=FALSE], 0) else
                            draws[, c(nontau, taunames), drop=FALSE]
         )
  }
}

##' Censored Ordinal Variable
##'
##' Creates a 2-column integer matrix that handles left- right- and interval-censored ordinal or continuous values for use in [blrm()].  A pair of values `[a, b]` represents an interval-censored value known to be in the interval `[a, b]` inclusive of `a` and `b`.  It is assumed that all distinct values are observed as uncensored for at least one observation.  When both input variables are `factor`s it is assume that the one with the higher number of levels is the one that correctly specifies the order of levels, and that the other variable does not contain any additional levels.  If the variables are not `factor`s it is assumed their original values provide the orderings.  Since all values that form the left or right endpoints of an interval censored value must be represented in the data, a left-censored point is is coded as `a=1` and a right-censored point is coded as `b` equal to the maximum observed value.  If the maximum observed value is not really the maximum possible value, everything still works except that predictions involving values above the highest observed value cannot be made.  As with most censored-data methods, [blrm()] assumes that censoring is independent of the response variable values that would have been measured had censoring not occurred.
##' @title Ocens
##' @param a vector representing a `factor`, numeric, or alphabetically ordered character strings
##' @param b like `a`.  If omitted, it copies `a`, representing nothing but uncensored values
##' @return a 2-column integer matrix of class `"Ocens"` with an attribute `levels` (ordered).  When the original variables were `factor`s, these are factor levels, otherwise are numerically or alphabetically sorted distinct (over `a` and `b` combined) values.  When the variables are not factors and are numeric, another attribute `median` is also returned.  This is the median of the uncensored values.  When the variables are factor or character, the median of the integer versions of variables for uncensored observations is returned as attribute `mid`.  A final attribute `freq` is the vector of frequencies of occurrences of all uncensored values.  `freq` aligns with `levels`.
##' @author Frank Harrell
##' @export
Ocens <- function(a, b=a) {
  nf <- is.factor(a) + is.factor(b)
  if(nf == 1)
    stop('one of a and b is a factor and the other is not')

  uncensored <- ! is.na(a) & ! is.na(b) & (a == b)
  if(! any(uncensored)) stop('no uncensored observations')
  ymed <- if(is.numeric(a)) median(a[uncensored])

  if(nf == 0) {
    num <- is.numeric(a) + is.numeric(b)
    if(num == 1) stop('one of a and b is numeric and the other is not')
    ## Since neither variable is a factor we can assume they are ordered
    ## numerics or use standard character string ordering
    ## Combine distinct values and create an nx2 matrix of integers
    u <- sort(unique(c(a, b)))
    u <- u[! is.na(u)]
    if(any(b < a)) stop('some values of b are less than corresponding a values')
    a    <- match(a, u)
    b    <- match(b, u)
    freq <- tabulate(a[uncensored], nbins=length(u))
    return(structure(cbind(a, b), class='Ocens', levels=u, freq=freq, median=ymed))
  }
  alev <- levels(a)
  blev <- levels(b)
  ## Cannot just pool the levels because ordering would not be preserved
  if(length(alev) >= length(blev)) {
    master <- alev
    other  <- blev
  } else{
    master <- blev
    other  <- alev
  }
  if(any(other %nin% master))
    stop('a variable has a level not found in the other variable')
  a <- match(as.character(a), master)
  b <- match(as.character(b), master)
  if(any(b < a)) stop('some values of b are less than corresponding a values')
  freq <- tabulate(a[uncensored], nbins=length(master))
  mid  <- quantile(a[uncensored], probs=.5, type=1L)
  structure(cbind(a, b), class='Ocens', levels=master, freq=freq, mid=mid)
}

##' Convert `Ocens` Object to Data Frame to Facilitate Subset
##'
##' Converts an `Ocens` object to a data frame so that subsetting will preserve all needed attributes
##' @title as.data.frame.Ocens
##' @param x an `Ocens` object
##' @param row.names optional vector of row names
##' @param optional set to `TRUE` if needed
##' @param ... ignored
##' @return data frame containing a 2-column integer matrix with attributes
##' @author Frank Harrell
##' @export
as.data.frame.Ocens <- function(x, row.names = NULL, optional = FALSE, ...) {
  nrows <- NROW(x)
  row.names <- if(optional) character(nrows) else as.character(1:nrows)
  value <- list(x)
  if(! optional) names(value) <- deparse(substitute(x))[[1]]
  structure(value, row.names=row.names, class='data.frame')
}
