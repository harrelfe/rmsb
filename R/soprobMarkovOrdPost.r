#' State Occupancy Probabilities for First-Order Markov Ordinal Model over Posterior Draws
#'
#' @title soprobMarkovOrdPost
#' @param y a vector of possible y values in order (numeric, character, factor)
#' @param times vector of measurement times
#' @param initial initial value of `y` (baseline state; numeric, character, factr)
#' @param absorb vector of absorbing states, a subset of `y`.  The default is no absorbing states. (numeric, character, factor)
#' @param intercepts vector of intercepts in the proportional odds model, with length one less than the length of `y`
#' @param ... additional arguments to pass to `g` such as covariate settings
#'
#' @return matrix with rows corresponding to times and columns corresponding to states, with values equal to exact state occupancy probabilities
#' @export
#' @author Frank Harrell
#' @seealso <https://hbiostat.org/R/Hmisc/markov/>
#' @export 
#' @md
# Compute unconditional probabilities
# Output is an array of dimension number of draws by s rows corresponding to
# t=times(1), times(2), ...
# times(s), and final dimension is the number of Y categories
# times(1) must be 1
# 
# fit is the Bayesian model fit from blrm which must contain the variable named day and the
# previous state, which must be named by the value of pvarname
# data is a data frame or data table that contains covariate settings
# other than day and what's in pvarname.  data has one row.
# times is the vector of days for which to evaluate state occupancy probabilities
# ylevels are the ordered levels of the outcome variable
# Set gap=TRUE to compute time gaps when predictor gap is in the model

soprobMarkovOrdPost <- function(object, data, times, ylevels,
                                tvarname='time', pvarname='yprev',
                                gap=NULL) {
	if(! length(object$draws)) stop('fit does not have posterior draws')
	nd <- nrow(object$draws)
  k  <- length(ylevels)
  s  <- length(times)
  P  <- array(NA, c(nd, s, k),
  						dimnames=list(paste('draw', 1 : nd), as.character(times), 
  													as.character(ylevels)))
  # Never uncondition on initial state
  data[[tvarname]] <- times[1]
  if(length(gap)) data[[gap]] <- times[1]
  p <- predict(object, data, type='fitted.ind', posterior.summary='all')
  P[, 1, ] <- p
  # cp: matrix of conditional probabilities of Y conditioning on previous time Y
  # Columns = k conditional probabilities conditional on a single previous state
  # Rows    = all possible previous states
  # This is for a single posterior draw
  cp <- matrix(NA, nrow=k, ncol=k, 
               dimnames=list(paste('t-1', ylevels), paste('t', ylevels)))
  data <- as.list(data)
  data[[pvarname]] <- as.character(ylevels[-1])    ### ??character?
  # don't request estimates after absorbing state
  edata <- expand.grid(data)
  for(it in 2 : s) {
    edata[[tvarname]] <- times[it]
    if(length(gap)) edata[[gap]] <- times[it] - times[it - 1]
    # y=first level = absorbing state
    pp <- predict(object, edata, type='fitted.ind', posterior.summary='all')
    for(idraw in 1 : nd) {
    	cp <- rbind(c(1., rep(0., k - 1)), pp[idraw, ,])
      # Compute unconditional probabilities of being in all possible states
      # at current time t
      P[idraw, it, ] <- t(cp) %*% P[idraw, it - 1, ]
    }
  }
  P
}

soprobMarkovOrd <- function(y, times, initial, absorb=NULL,
                            intercepts, g, ...) {

  if(initial %in% absorb) stop('initial state cannot be an absorbing state')
  k  <- length(y)
  nt <- length(times)
  P  <- matrix(NA, nrow=nt, ncol=k)
  colnames(P) <- as.character(y)
  rownames(P) <- as.character(times)
  yna         <- setdiff(y, absorb)   # all states except absorbing ones
  yc          <- as.character(y)
  ynac        <- as.character(yna)

  ## Don't uncondition on initial state
  xb <- g(initial, times[1], times[1], ...)  # 3rd arg (gap) assumes time origin 0
  ## Since initial is scalar, xb has one row.  It has multiple columns if
  ## model is partial PO model, with columns exactly corresponding to intercepts
  pp <- plogis(intercepts + xb)
  ## Compute cell probabilities
  pp <- c(1., pp) - c(pp, 0.)
  P[1, ] <- pp

  tprev <- times[1]
  for(it in 2 : nt) {
    t <- times[it]
    gap <- t - tprev
    ## Compute linear predictor at all non-absorbing states
    xb <- g(yna, t, gap, ...)   #non-intercept part of x * beta
    ## g puts non-absorbing states as row names (= ynac)
    ## If partial PO model xb has > 1 column that correspond to intercepts

    ## Matrix of conditional probabilities of Y conditioning on previous Y
    ## Columns = k conditional probabilities conditional on a single previous state
    ## Rows    = all possible previous states
    ## When the row corresponds to an absorbing state, with prob. 1
    ## a subject will remain in that state so give it a prob of 1 and
    ## all other states a prob of 0

    cp <- matrix(NA, nrow=k, ncol=k, dimnames=list(yc, yc))
    for(yval in y) {   # current row
      yvalc <- as.character(yval)
      if(yval %in% absorb) {   # current row is an absorbing state
        cp[yvalc, setdiff(yc, yvalc)]    <- 0.  # P(moving to non-abs state)=0
        cp[yvalc, yvalc]                 <- 1.  # certainty in staying
      }
      else {  # current row is non-absorbing state
        pp <- plogis(intercepts + xb[yvalc, ])
        ## Compute cell probabilities
        pp <- c(1., pp) - c(pp, 0.)
        cp[yvalc, ] <- pp
      }
    }
    P[it, ] <- t(cp) %*% P[it - 1, ]
    tprev <- t
  }
  P
}
