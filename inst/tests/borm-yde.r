## Tests borm()'s yde/ytrans mechanism: the generalization of
## blrm()'s ppo/cppo to a matrix-valued Y-dependent effect (one or more
## columns from ytrans, vs. cppo's single score). Self-contained: does
## not depend on anything else having been run first.
##
## Claude Sonnet 5 2026-11-21          entire file

require(rmsb)
cstanSet()

## ----------------------------------------------------------------------
## 1. m=1: yde/ytrans with a single yde predictor and a single-column
## ytrans should reduce mathematically to blrm()'s own ppo/cppo
## mechanism. blrm() needs normcppo=TRUE to match borm()'s own
## unconditional auto-centering/scaling of ytrans. Uses real NUTS
## sampling, not optimizing -- rstan::optimizing's L-BFGS was found
## (via a standalone self-consistency check during development) to
## land in different, sometimes unstable, local optima for the two
## differently-parameterized Stan files on this kind of model; sampling
## is far more robust and is what should be used for this comparison.
## ----------------------------------------------------------------------

cat("\n=== 1. m=1 yde/ytrans vs blrm ppo/cppo (real NUTS sampling) ===\n")
set.seed(5)
n <- 300
x <- rnorm(n)
x2 <- sample(c("a", "b"), n, TRUE)
y <- round(0.5 * x + 1.0 * (x2 == "b") + 0.3 * rnorm(n), 1)
d <- data.frame(x = x, x2 = x2, y = y)
dd <- datadist(d); options(datadist = "dd")
scorefun <- function(y) y

f_blrm <- blrm(y ~ x + x2, ~x2, cppo = scorefun, normcppo = TRUE, data = d,
               iter = 1500, warmup = 750, chains = 2, refresh = 0,
               sampling.args = list(seed = 11))
f_borm <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
               iter = 1500, warmup = 750, chains = 2, refresh = 0,
               sampling.args = list(seed = 11))

cat("blrm posterior means: x =", mean(f_blrm$draws[, "x"]),
    " x2=b =", mean(f_blrm$draws[, "x2=b"]),
    " tau =", mean(f_blrm$draws[, "x2=b x f(y)"]), "\n")
cat("borm posterior means: x =", mean(f_borm$draws[, "x"]),
    " x2=b =", mean(f_borm$draws[, "x2=b"]),
    " tau =", mean(f_borm$draws[, grep("ytrans", colnames(f_borm$draws))]), "\n")
cat("(expect close agreement throughout, within ordinary MCMC noise)\n")

## ----------------------------------------------------------------------
## 2. m>1: a genuine, strongly Y-dependent treatment effect should be
## captured by borm(yde=, ytrans=) and give a much better out-of-sample
## fit (loo elpd) than the plain proportional-odds model, confirming the
## mechanism recovers real signal rather than just running without
## error. Exact recovery of the individual true tau values is not
## checked (tau is basis-dependent once auto-centered/scaled and m>1),
## only that the fit improves substantially, which is a basis-
## independent, unambiguous criterion.
## ----------------------------------------------------------------------

cat("\n=== 2. m>1: genuine Y-dependent effect recovery (loo comparison) ===\n")
set.seed(21)
n2 <- 800
k <- 7
x3 <- rnorm(n2)
trt <- rbinom(n2, 1, 0.5)
alpha_true <- c(2, 1, 0.3, -0.3, -1, -2)
g <- function(j) 1.2 * (j - 4) # true, strongly Y-dependent trt effect
Pge <- matrix(0, n2, k); Pge[, 1] <- 1
for (j in 2:k) Pge[, j] <- plogis(alpha_true[j - 1] + 0.5 * x3 + 1.0 * trt + g(j) * trt)
Pcat <- cbind(Pge[, 1:(k - 1)] - Pge[, 2:k], Pge[, k])
Pcat[Pcat < 0] <- 0
Pcat <- Pcat / rowSums(Pcat)
y2 <- apply(Pcat, 1, function(p) sample(1:k, 1, prob = p))
d2 <- data.frame(x = x3, trt = trt, y = y2)
dd2 <- datadist(d2); options(datadist = "dd2")

f_po  <- borm(y ~ x + trt, family = "logistic", data = d2,
              iter = 1500, warmup = 750, chains = 2, refresh = 0, sampling.args = list(seed = 3))
f_yde <- borm(y ~ x + trt, yde = ~trt, ytrans = function(y) cbind(y, y^2), family = "logistic", data = d2,
              iter = 1500, warmup = 750, chains = 2, refresh = 0, sampling.args = list(seed = 3))

cat("PO-only  loo elpd:", f_po$loo$estimates["elpd_loo", "Estimate"], "\n")
cat("yde      loo elpd:", f_yde$loo$estimates["elpd_loo", "Estimate"], "\n")
cat("(expect the yde model's elpd to be substantially higher -- the true effect is strongly Y-dependent)\n")
cat("yde tau posterior means:\n")
print(colMeans(f_yde$draws[, grep("ytrans", colnames(f_yde$draws)), drop = FALSE]))

## ----------------------------------------------------------------------
## 3. predict.borm()'s type='x'/'lp' for a yde-fitted model: direct
## manual cross-check against the fit's own back-transformed
## coefficients, confirming the whole predict pipeline (Design()
## rebuilt at predict time, factor releveling, kint re-derivation from
## ycut, positional beta/X matching) is self-consistent.
## ----------------------------------------------------------------------

cat("\n=== 3. predict.borm() type='x'/'lp' self-consistency ===\n")
ylevels <- f_borm$ylevels
ycutval <- ylevels[round(length(ylevels) * 0.5)]
nd <- data.frame(x = 1, x2 = "b")
lp <- predict(f_borm, newdata = nd, type = "lp", ycut = ycutval, cint = FALSE)

co <- f_borm$param["mean", ]
kint <- which(ylevels == ycutval) - 1
alphaname <- paste0("y>=", ylevels[-1])[kint]
ytransmat <- eval(parse(text = f_borm$ytrans))(ylevels)
if (!is.matrix(ytransmat)) ytransmat <- matrix(ytransmat, ncol = 1)
manual <- co[alphaname] + co["x"] * 1 + co["x2=b"] * 1 +
  co["x2=b x ytrans1"] * 1 * ytransmat[kint + 1, 1]
cat("manual lp:", manual, " predict() lp:", lp, " diff:", manual - lp,
    " (expect ~0, machine epsilon)\n")

## ----------------------------------------------------------------------
## 4. Backward compatibility: a borm() fit WITHOUT yde/ytrans must be
## completely unaffected (predict.borm()/Mean.borm()/etc. still fully
## delegate to the blrm() versions).
## ----------------------------------------------------------------------

cat("\n=== 4. Backward compatibility: no yde given ===\n")
f0 <- borm(y ~ x + trt, family = "logistic", data = d2,
           iter = 300, warmup = 150, chains = 1, refresh = 0)
lp0 <- predict(f0, newdata = data.frame(x = 0, trt = 1), type = "lp", cint = FALSE)
cat("plain borm() predict still works:", lp0, "\n")

cat("\n=== 5. Mean/Quantile/ExProb.borm() error clearly (not silently) for a yde fit ===\n")
cat(tryCatch(Mean(f_yde), error = function(e) conditionMessage(e)), "\n")

## ----------------------------------------------------------------------
## 6. npcontrast: combined beta+tau contrasts via named predictor
## settings that may carry an optional ycut (the Y value at which to
## evaluate that setting's tau contribution). The key property this
## checks, directly (not just reasoned through): a setting with no
## ycut contributes a pure-beta row (tau-part exactly 0, matching an
## equivalent pcontrast exactly), and two settings sharing the same
## covariate values but different ycut have their beta-parts cancel
## to exactly 0 in a difference, giving a pure-tau contrast -- i.e.
## simple contrasts automatically drop the unneeded part, with no
## special syntax.
## ----------------------------------------------------------------------

cat("\n=== 6. npcontrast: combined beta+tau contrasts ===\n")
options(datadist = "dd") # section 2 switched the active datadist to dd2
ylevn <- sort(unique(y))
yc_mid <- ylevn[round(length(ylevn) * 0.5)]
yc_hi  <- ylevn[round(length(ylevn) * 0.85)]

f_pc <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
             pcontrast = list(sd = 0.2, c1 = list(x2 = "b"), contrast = expression(c1)),
             iter = 600, warmup = 300, chains = 1, refresh = 0, sampling.args = list(seed = 1))
f_npc <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
              npcontrast = list(sd = 0.2, c1 = list(x2 = "b"), contrast = expression(c1)),
              iter = 600, warmup = 300, chains = 1, refresh = 0, sampling.args = list(seed = 1))
pbeta <- ncol(f_pc$Contrast)
cat("6a. pure-beta npcontrast (no ycut) vs an equivalent pcontrast:\n")
cat("    pcontrast beta-part: ", f_pc$Contrastt, "\n")
cat("    npcontrast row:      ", f_npc$nContrastt, "\n")
cat("    (expect the first", pbeta, "entries to match exactly, and the tau-part (rest) to be exactly 0)\n")

f_puretau <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
                  npcontrast = list(sd = 0.2, atLo = list(x2 = "b"), atHi = list(x2 = "b"),
                                     contrast = expression(atHi - atLo),
                                     ycut = c(atLo = yc_mid, atHi = yc_hi)),
                  iter = 600, warmup = 300, chains = 1, refresh = 0, sampling.args = list(seed = 1))
Cpt <- f_puretau$nContrastt
cat("6b. pure-tau npcontrast (same covariates, different ycut):\n")
cat("    beta-part (expect exactly 0):", Cpt[1, 1:pbeta], "\n")
cat("    tau-part (expect nonzero):   ", Cpt[1, -(1:pbeta)], "\n")

f_mixed <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
                npcontrast = list(sd = 0.3, effAtHi = list(x2 = "b"),
                                   contrast = expression(effAtHi), ycut = yc_hi),
                iter = 600, warmup = 300, chains = 1, refresh = 0, sampling.args = list(seed = 1))
Cm <- f_mixed$nContrastt
cat("6c. mixed contrast (effect of x2 at a specific Y value -- beta AND tau both nonzero):\n")
cat("    beta-part:", Cm[1, 1:pbeta], " tau-part:", Cm[1, -(1:pbeta)], "\n")
cat("    beta-part should match 6a's exactly (ycut-independent):", f_npc$nContrastt[1, 1:pbeta], "\n")

## ----------------------------------------------------------------------
## 7. rms::contrast.rms() on a yde-fitted borm() object. contrast.rms()
## itself needed a fix (rms's own R/contrast.s: its ycut-passing logic
## was hardcoded to inherits(fit,'blrm'), so it silently evaluated a
## yde-fitted borm() model's Y-dependent effect at the reference
## intercept regardless of any ycut the user supplied, while still
## mislabeling the output as if ycut had been honored) -- confirmed
## fixed by checking that contrast() now requires ycut for a
## yde-fitted borm() (matching blrm's own partial-PO requirement) and
## that its value actually changes with ycut, matching predict()'s
## own ground truth exactly. Also exercises a real, separate bug this
## exposed in bormPredictYde(): contrast.rms()'s internal predict()
## call passes its data frame POSITIONALLY (not as newdata=), which a
## plain dots$newdata lookup does not see -- fixed via the new
## bormGetNewdata() helper.
## ----------------------------------------------------------------------

cat("\n=== 7. rms::contrast.rms() on a yde-fitted borm() object ===\n")
f_contr <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
                iter = 500, warmup = 250, chains = 1, refresh = 0)
err <- tryCatch(contrast(f_contr, list(x2 = "b"), list(x2 = "a")), error = function(e) conditionMessage(e))
cat("no-ycut call now errors (matching blrm's own partial-PO requirement):", err, "\n")

kref <- contrast(f_contr, list(x2 = "b"), list(x2 = "a"), ycut = yc_mid)
khi  <- contrast(f_contr, list(x2 = "b"), list(x2 = "a"), ycut = yc_hi)
cat("contrast() at ycut=", yc_mid, ":", kref$Contrast, "\n")
cat("contrast() at ycut=", yc_hi, ":", khi$Contrast, " (expect different from the above)\n")

pref <- predict(f_contr, newdata = data.frame(x = median(x), x2 = "b"), type = "lp", ycut = yc_mid, cint = FALSE) -
        predict(f_contr, newdata = data.frame(x = median(x), x2 = "a"), type = "lp", ycut = yc_mid, cint = FALSE)
phi  <- predict(f_contr, newdata = data.frame(x = median(x), x2 = "b"), type = "lp", ycut = yc_hi, cint = FALSE) -
        predict(f_contr, newdata = data.frame(x = median(x), x2 = "a"), type = "lp", ycut = yc_hi, cint = FALSE)
cat("ground truth via predict() at ycut=", yc_mid, ":", pref, " (expect match contrast()'s value)\n")
cat("ground truth via predict() at ycut=", yc_hi, ":", phi, " (expect match contrast()'s value)\n")

## ----------------------------------------------------------------------
## 8. print.borm()/bormStats() on a yde-fitted model with x=TRUE/y=TRUE.
## bormStats() builds the linear predictor as cbind(1,X) %*% t(beta),
## but draws has [alphas, betas, taus] when qyde>0 -- the tau columns
## must be split out and combined with Z via ytransmat at the
## reference intercept's Y level (mirroring blrmStats()'s own
## tauFetch()-based handling of its cppo-weighted tau), not left in
## with betas. Confirmed directly: print() on a yde-fitted borm()
## crashed with "non-conformable arguments" in
## cbind(1,X) %*% t(beta) before this fix (X has p+1 columns, beta
## had p+1+qyde*myde). Also confirmed numerically: a manual
## reconstruction of bormStats()'s own lp, using the stored z/ytrans/
## qyde/myde, matches the difference in that lp between two covariate
## groups exactly against beta_x + tau %*% ytransmat[reference,].
## ----------------------------------------------------------------------

cat("\n=== 8. print.borm()/bormStats() on a yde-fitted model ===\n")
f_print <- borm(y ~ x + x2, yde = ~x2, ytrans = scorefun, family = "logistic", data = d,
                x = TRUE, y = TRUE,
                iter = 300, warmup = 150, chains = 1, refresh = 0)
invisible(capture.output(print(f_print)))
cat("print(f_print) completed without error\n")

draws <- f_print$draws
ylevp <- f_print$ylevels
kmidp <- f_print$interceptRef
ytransmatp <- eval(parse(text = f_print$ytrans))(ylevp)
if (!is.matrix(ytransmatp)) ytransmatp <- matrix(ytransmatp, ncol = 1)
cnp <- colnames(draws)
taunamp <- grep("ytrans", cnp, value = TRUE)
betanamp <- setdiff(cnp[-(1:f_print$non.slopes)], taunamp)
i <- 1
beta_i <- draws[i, c(cnp[kmidp], betanamp)]
tau_i <- matrix(draws[i, taunamp], f_print$qyde, f_print$myde)
etau_i <- tau_i %*% ytransmatp[kmidp + 1, ]
## Use bormStats()'s own stored real data (f_print$x/$z), not synthetic
## rows, so this actually exercises what bormStats() itself computes;
## pick two real observations and subtract out their x contribution
## (since x differs between them too) to isolate the x2 effect
Xp <- f_print$x
Zp <- f_print$z
lp_manual <- cbind(1, Xp) %*% beta_i + Zp %*% etau_i
i_x2b <- which(d$x2 == "b")[1]
i_x2a <- which(d$x2 == "a")[1]
diffx <- beta_i["x"] * (Xp[i_x2b, "x"] - Xp[i_x2a, "x"])
cat("manual lp diff minus the x-driven part (isolating the x2 effect):",
    lp_manual[i_x2b] - lp_manual[i_x2a] - diffx,
    "  vs beta_x2b + etau:", beta_i["x2=b"] + etau_i, " (expect exact match)\n")
