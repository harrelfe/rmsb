// For the future rmsb::borm() (Bayesian analog of rms::orm()): general-
// link-family ordinal regression with a Dirichlet prior on the
// intercepts (as in lrmconppo.stan) and the precomputed ia/ia2/sgn
// intercept-subscripting method from rms::ormIdx() (see lrmconppo.stan
// for the full rationale: it removes all censoring-type conditionals
// from the Stan code). This file generalizes that approach three ways:
//
// Claude Sonnet 5 2026-11-17          entire file
//
// 1. All 5 cumulative-probability link families rms::orm() supports
//    (logistic, probit, loglog, cloglog, cauchit) are implemented,
//    encoded as integer link = 1..5 in data. link is fixed for an
//    entire fit (one borm() call always uses one family), so checking
//    it inside the per-observation (1:N) loop would re-evaluate the
//    same branch N times for nothing. Instead there is one
//    pw_log_lik_<family> function per family, each with an identical
//    loop that never branches on link at all, and a single dispatcher
//    function that branches on link exactly once (per log_prob
//    evaluation, not per observation) to pick which of the 5 to call.
// 2. pw_log_lik takes the linear predictor lp = X*beta (computed once,
//    vectorized, in transformed parameters) instead of X and beta
//    directly. X*beta as a single matrix-vector product is computed
//    far more efficiently by Stan's autodiff than the N separate
//    row_vector dot products a per-observation X[n]*beta would need
//    (one small autodiff expression per observation vs. one large,
//    well-vectorized one) -- the same reasoning as point 1, applied to
//    a different loop-invariant-structure cost. lp2 is a placeholder,
//    always 0 and passed as plain `data` from R for now, for the
//    future ydep Y-dependent linear predictor contribution (see
//    CLAUDE.md's ydep notes). Once ydep has real coefficients to
//    estimate, lp2 can no longer be `data`: like lp = X*beta above,
//    any expression combining a parameter with data must be computed
//    inside Stan (transformed parameters, or inline), not precomputed
//    in R, so that autodiff sees the dependency -- the fixed Y-
//    dependent score design can still come from R as data, same as
//    blrm's pposcore/Z, but its combination with the new ydep
//    coefficient parameter(s) has to happen here. Separately, ydep's
//    exact per-alpha-index indexing (parallel to blrm's
//    pposcore[ia+1] vs pposcore[ia2+1] -- a real Y-dependent effect
//    varies by which of ia/ia2 is being evaluated, not just by
//    observation) is an open design question, not yet settled with
//    Frank; a single vector[N] lp2 shared identically between e1 and
//    e2, as used below, is almost certainly the wrong shape once that
//    is designed and will need revisiting then, even though it is a
//    harmless placeholder while it is always 0.
// 3. rms::orm.rfit()'s dual-sigma mix_re extension to the random
//    intercept -- gamma_ij = [sigma1*(1-mre_ij) + sigma2*mre_ij] * v_i,
//    sigma1 constrained positive, sigma2 free -- is implemented
//    alongside the ordinary single-sigma cluster/gamma mechanism blrm
//    already uses (reused here via the same gamma_raw/cluster
//    machinery), selected by the use_mre data flag. mre is fixed, user-
//    supplied data, exactly as in orm.fit's own mre argument -- never
//    estimated. As in points 1-2, the per-observation random-effect
//    contribution (gamma_eta) is computed once, vectorized, in
//    transformed parameters -- via vector[Nc] gamma_raw gathered by
//    cluster into a vector[N], a single elementwise multiply -- so
//    pw_log_lik's own loops branch on neither link nor use_mre.
//
// The only genuinely per-observation conditions left in the inner
// loops are ia2[n] > 0 (two-component vs. one-component likelihood)
// and the sign of sgn[n] -- both are data-dependent per observation
// (censoring legitimately differs row to row) and cannot be hoisted.
//
// Claude Sonnet 5 2026-11-17          entire paragraph
// logistic, loglog, and cloglog don't even need that per-observation
// loop: their log-CDF/CCDF forms above are built entirely from Stan's
// elementwise-vectorized math functions (log_inv_logit, log1m_exp,
// exp, log_inv_logit_diff) rather than its summing distribution
// functions (_lcdf/_lccdf sum a vector argument to a single real, by
// Stan convention -- confirmed directly: std_normal_lcdf(vector)
// compiles only as a scalar return). So for those three families
// pw_log_lik_* is a handful of vector expressions with no loop and no
// per-observation branch at all, selecting between the one- and two-
// component formula via a 0/1 indicator vector (has2) and arithmetic
// blending instead -- Stan has no elementwise ternary on a vector
// condition (confirmed directly: stanc rejects it, requiring a scalar
// int). Both the intercept index (ia2_safe) AND the resulting value
// (e2) fed to the unused branch have to be clamped to something valid
// on every row, not just ia2_safe: the unused branch's own domain
// violation (e1 > e2 not holding) produces -Inf/NaN that 0 .* (...)
// does NOT zero out in IEEE arithmetic (confirmed directly: 0*(-Inf)
// is NaN, not 0) -- see pw_log_lik_logistic's own comment for the fix.
// probit and cauchit are NOT vectorized this way: their
// numerically-stable forms are std_normal_lcdf/cauchy_lcdf, which
// only exist in the summing form, so there is no accuracy-preserving
// way to get a per-element vector from them directly; the available
// elementwise alternative (log(Phi(x)) / log(0.5+atan(x)/pi)) would
// reintroduce the same kind of tail cancellation std_normal_lcdf/
// cauchy_lcdf exist to avoid, unmeasured here, so they keep the
// scalar per-observation loop instead of trading accuracy for speed.
//
// No priors, contrasts, or Stan code for ppo/cppo are included: borm
// does not have them yet (ydep, when designed, will play that role).
// No pcontrast support either -- not requested for borm.

functions {
  // ---- per-family stable log CDF / log complementary CDF ----
  // Reuses Stan's own built-in, already log-scale-stable CDF functions
  // wherever one exists. logistic, probit, and cauchit are symmetric
  // about 0 (1-F(x) = F(-x)), so no separate complementary-CDF
  // function is needed for them -- log CCDF is just log CDF of -x.
  // loglog and cloglog are NOT symmetric (they are reflections of EACH
  // OTHER, not of themselves: 1-F_loglog(x) = F_cloglog(-x)), so their
  // complementary forms are written out explicitly; both of these, and
  // both of their ordinary CDFs, are exact closed forms (one exp()
  // each, or one log1m_exp() of an already-negative argument), so no
  // cancellation is possible in any of the four.
  real logCDF_logistic (real x) { return log_inv_logit(x); }
  real logCCDF_logistic(real x) { return log_inv_logit(-x); }
  real logCDF_probit   (real x) { return std_normal_lcdf(x); }
  real logCCDF_probit  (real x) { return std_normal_lcdf(-x); }
  real logCDF_cauchit  (real x) { return cauchy_lcdf(x | 0, 1); }
  real logCCDF_cauchit (real x) { return cauchy_lcdf(-x | 0, 1); }
  real logCDF_loglog   (real x) { return -exp(-x); }               // exact
  real logCCDF_loglog  (real x) { return log1m_exp(-exp(-x)); }
  real logCDF_cloglog  (real x) { return log1m_exp(-exp(x)); }
  real logCCDF_cloglog (real x) { return -exp(x); }                // exact
  // Claude Sonnet 5 2026-11-17          4 lines
  // Vector overloads (same formulas, built entirely from elementwise-
  // vectorized primitives) so pw_log_lik_loglog/cloglog below can be
  // fully vectorized -- see the file header note on why this isn't
  // done for probit/cauchit.
  vector logCDF_loglog   (vector x) { return -exp(-x); }
  vector logCCDF_loglog  (vector x) { return log1m_exp(-exp(-x)); }
  vector logCDF_cloglog  (vector x) { return log1m_exp(-exp(x)); }
  vector logCCDF_cloglog (vector x) { return -exp(x); }

  // ---- logistic/loglog/cloglog: fully vectorized, no loop at all ----
  // e is the same quantity (lp + lp2 + random-effect contribution) for
  // both the p1 and p2 components of one observation; only the
  // intercept differs between them. ia2_safe is ia2 with every 0
  // (meaning "no second component") clamped to a dummy valid index
  // (1) so alpha[ia2_safe] never goes out of bounds; has2 (1 where
  // ia2 > 0, 0 otherwise) then selects, by elementwise arithmetic
  // blending, between the two-component and one-component result --
  // both are computed for every observation (cheap, pure vector ops)
  // and the unwanted one is multiplied away, rather than branching.
  // Both ia2_safe and has2 are precomputed once in transformed data,
  // not here, since this function runs once per log_prob evaluation.
  //
  // Claude Sonnet 5 2026-11-17          3 lines
  // e2 (not just ia2_safe) must itself be clamped, to e1 - 1, whenever
  // has2 == 0: alpha[ia2_safe]+e is a real but ARBITRARY value there
  // (the dummy index 1 has no relation to ia[n]), so log_inv_logit_diff/
  // log_diff_exp's required e1 > e2 need not hold on an unused row --
  // confirmed directly: a stray unused-row violation gives -Inf or NaN,
  // and 0 .* that is itself NaN in IEEE arithmetic (0 * Inf = NaN), not
  // 0, silently corrupting every row near it despite has2 being 0.
  // e1 - 1 guarantees e1 > e2 on every row regardless of has2, with no
  // ternary needed (has2 blends it away exactly where it doesn't apply).
  vector pw_log_lik_logistic(vector alpha, vector lp, vector lp2, vector gamma_eta,
                             array[] int ia, array[] int ia2_safe, vector has2, vector sgn) {
    vector[num_elements(lp)] e  = lp + lp2 + gamma_eta;
    vector[num_elements(lp)] e1 = alpha[ia] + e;
    vector[num_elements(lp)] e2 = has2 .* (alpha[ia2_safe] + e) + (1 - has2) .* (e1 - 1);
    return has2 .* log_inv_logit_diff(e1, e2) + (1 - has2) .* log_inv_logit(sgn .* e1);
  }

  vector pw_log_lik_probit(vector alpha, vector lp, vector lp2, vector gamma_eta,
                           array[] int ia, array[] int ia2, vector sgn) {
    int N = num_elements(lp);
    vector[N] out;
    for (n in 1:N) {
      real e  = lp[n] + lp2[n] + gamma_eta[n];
      real e1 = alpha[ia[n]] + e;
      if (ia2[n] > 0) out[n] = log_diff_exp(logCDF_probit(e1), logCDF_probit(alpha[ia2[n]] + e));
      else            out[n] = sgn[n] > 0 ? logCDF_probit(e1) : logCCDF_probit(e1);
    }
    return out;
  }

  vector pw_log_lik_loglog(vector alpha, vector lp, vector lp2, vector gamma_eta,
                           array[] int ia, array[] int ia2_safe, vector has2, vector sgn) {
    vector[num_elements(lp)] e  = lp + lp2 + gamma_eta;
    vector[num_elements(lp)] e1 = alpha[ia] + e;
    // See pw_log_lik_logistic for why e2 itself (not just ia2_safe)
    // must be clamped on unused (has2 == 0) rows.
    vector[num_elements(lp)] e2 = has2 .* (alpha[ia2_safe] + e) + (1 - has2) .* (e1 - 1);
    // sgn > 0 selects logCDF vs. logCCDF elementwise; sgn is +-1, so
    // (1+sgn)/2 is a 0/1 indicator without needing a comparison.
    vector[num_elements(lp)] indF = (1 + sgn) / 2;
    return has2 .* log_diff_exp(logCDF_loglog(e1), logCDF_loglog(e2))
         + (1 - has2) .* (indF .* logCDF_loglog(e1) + (1 - indF) .* logCCDF_loglog(e1));
  }

  vector pw_log_lik_cloglog(vector alpha, vector lp, vector lp2, vector gamma_eta,
                            array[] int ia, array[] int ia2_safe, vector has2, vector sgn) {
    vector[num_elements(lp)] e  = lp + lp2 + gamma_eta;
    vector[num_elements(lp)] e1 = alpha[ia] + e;
    vector[num_elements(lp)] e2 = has2 .* (alpha[ia2_safe] + e) + (1 - has2) .* (e1 - 1);
    vector[num_elements(lp)] indF = (1 + sgn) / 2;
    return has2 .* log_diff_exp(logCDF_cloglog(e1), logCDF_cloglog(e2))
         + (1 - has2) .* (indF .* logCDF_cloglog(e1) + (1 - indF) .* logCCDF_cloglog(e1));
  }

  vector pw_log_lik_cauchit(vector alpha, vector lp, vector lp2, vector gamma_eta,
                            array[] int ia, array[] int ia2, vector sgn) {
    int N = num_elements(lp);
    vector[N] out;
    for (n in 1:N) {
      real e  = lp[n] + lp2[n] + gamma_eta[n];
      real e1 = alpha[ia[n]] + e;
      if (ia2[n] > 0) out[n] = log_diff_exp(logCDF_cauchit(e1), logCDF_cauchit(alpha[ia2[n]] + e));
      else            out[n] = sgn[n] > 0 ? logCDF_cauchit(e1) : logCCDF_cauchit(e1);
    }
    return out;
  }

  // The only place that branches on link -- once per log_prob
  // evaluation, not once per observation. ia2 is the raw subscript
  // (needed by the scalar-loop probit/cauchit branches); ia2_safe/
  // has2 are the precomputed vectorized-selection pair (see transformed
  // data) needed by the vectorized logistic/loglog/cloglog branches.
  vector pw_log_lik(int link, vector alpha, vector lp, vector lp2, vector gamma_eta,
                    array[] int ia, array[] int ia2, array[] int ia2_safe, vector has2,
                    vector sgn) {
    if      (link == 1) return pw_log_lik_logistic(alpha, lp, lp2, gamma_eta, ia, ia2_safe, has2, sgn);
    else if (link == 2) return pw_log_lik_probit   (alpha, lp, lp2, gamma_eta, ia, ia2, sgn);
    else if (link == 3) return pw_log_lik_loglog   (alpha, lp, lp2, gamma_eta, ia, ia2_safe, has2, sgn);
    else if (link == 4) return pw_log_lik_cloglog  (alpha, lp, lp2, gamma_eta, ia, ia2_safe, has2, sgn);
    else                 return pw_log_lik_cauchit  (alpha, lp, lp2, gamma_eta, ia, ia2, sgn);
  }
}

data {
  int<lower = 1> N;                 // number of observations
  int<lower = 1> p;                 // number of predictors
  int<lower = 2> k;                 // number of outcome categories
  int<lower = 1, upper = 5> link;   // 1=logistic 2=probit 3=loglog 4=cloglog 5=cauchit
  matrix[N, p] X;                   // matrix of CENTERED predictors
  vector[N] lp2;                    // placeholder for future ydep contribution; 0 for now
  // Precomputed via rms::ormIdx() in R; see lrmconppo.stan.
  array[N] int<lower = 1, upper = k - 1> ia;  // intercept index for p1
  array[N] int<lower = 0, upper = k - 1> ia2; // intercept index for p2 (0 = none)
  vector[N] sgn;                              // +1 if p1 is F(), -1 if 1-F()

  real<lower = 0> conc;              // Dirichlet concentration parameter for intercepts

  int<lower = 0> Nc;                  // number of clusters (0 = no clustering)
  array[Nc == 0 ? 0 : N] int<lower = 1, upper = Nc> cluster;
  // Claude Sonnet 5 2026-11-17          2 lines
  // mix_re: rms::orm.rfit()'s dual-sigma random-intercept extension.
  int<lower = 0, upper = 1> use_mre;  // 1 = dual-sigma mix_re model (requires Nc > 0)
  vector[Nc == 0 || use_mre == 0 ? 0 : N] mre;  // fixed per-observation mix_re weight

  int<lower = 1, upper = 2> psigma;  // 1=t(4, rsdmean[1], rsdsd[1]), 2=exponential
  array[Nc == 0 ? 0 : 1] real<lower = 0> rsdmean;
  // Claude Sonnet 5 2026-11-17          2 lines
  // rsdsd is needed whenever psigma==1 (sigmag's or sigma1's half-t
  // prior), and ALSO whenever use_mre==1 (sigma2's prior below reuses
  // it as a scale regardless of psigma) -- flagged for review.
  array[Nc == 0 || (psigma == 2 && use_mre == 0) ? 0 : 1] real<lower = 0> rsdsd;
}

transformed data {
  // Claude Sonnet 5 2026-11-17          entire block
  // One-time (not per log_prob evaluation) setup for the vectorized
  // logistic/loglog/cloglog branches of pw_log_lik: ia2_safe is ia2
  // with 0 ("no second component") clamped to 1 so alpha[ia2_safe]
  // is always a valid index, and has2 is the corresponding 0/1
  // indicator used to blend the one- and two-component results. A
  // loop here costs nothing -- transformed data runs once per chain,
  // not once per gradient evaluation like transformed parameters.
  array[N] int ia2_safe;
  vector[N] has2;
  for (n in 1:N) {
    ia2_safe[n] = max(ia2[n], 1);
    has2[n] = ia2[n] > 0 ? 1 : 0;
  }
}

parameters {
  vector[p] beta;                    // coefficients on X
  simplex[k] pi;                      // category probabilities at X = 0 (before centering)
  vector[Nc] gamma_raw;                // unscaled per-cluster random effects
  // Exactly one of sigmag or (sigma1, sigma2) is actually used, chosen
  // by use_mre; the other is declared with 0 elements (same idiom
  // blrm's Stan code already uses for sigmag itself when Nc == 0).
  array[Nc == 0 || use_mre == 1 ? 0 : 1] real<lower = 0> sigmag;    // single-sigma SD
  array[Nc == 0 || use_mre == 0 ? 0 : 1] real<lower = 0> sigma1;    // mix_re SD at mre = 0
  array[Nc == 0 || use_mre == 0 ? 0 : 1] real sigma2;               // mix_re SD at mre = 1 (sign free)
}

transformed parameters {
  vector[k - 1] alpha;
  vector[N] lp = X * beta;            // one vectorized matrix-vector product, not N dot products
  vector[N] gamma_eta;                 // precomputed per-observation random-effect contribution
  vector[N] log_lik;

  for (j in 2:k) alpha[j - 1] = logit(sum(pi[j:k])); // predictors are CENTERED

  if (Nc == 0) {
    gamma_eta = rep_vector(0, N);
  } else if (use_mre == 0) {
    gamma_eta = sigmag[1] * gamma_raw[cluster];
  } else {
    gamma_eta = (sigma1[1] * (1 - mre) + sigma2[1] * mre) .* gamma_raw[cluster];
  }

  log_lik = pw_log_lik(link, alpha, lp, lp2, gamma_eta, ia, ia2, ia2_safe, has2, sgn);
}

model {
  if (Nc > 0) {
    gamma_raw ~ std_normal();  // implies gamma ~ normal(0, sigma) per cluster
    if (use_mre == 0) {
      if (psigma == 1) sigmag ~ student_t(4, rsdmean[1], rsdsd[1]);
      else             sigmag ~ exponential(1. / rsdmean[1]);
    } else {
      if (psigma == 1) sigma1 ~ student_t(4, rsdmean[1], rsdsd[1]);
      else             sigma1 ~ exponential(1. / rsdmean[1]);
      // Claude Sonnet 5 2026-11-17          1 line
      // sigma2's sign is free, so the half-t/exponential duality above
      // (for a positive-constrained SD) doesn't apply to it. A plain
      // normal(0, rsdsd) is a reasonable default, not something
      // rms::orm.rfit() specifies (it's an MLE, with no prior at all);
      // flagged for Frank's review rather than treated as settled.
      sigma2 ~ normal(0, rsdsd[1]);
    }
  }
  target += log_lik;
  target += dirichlet_lpdf(pi | rep_vector(conc, k));
}
