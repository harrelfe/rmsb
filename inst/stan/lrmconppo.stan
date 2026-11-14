// For constrained partial PO model with Dirichlet prior for transformed intercepts

// Claude Sonnet 5 2026-11-14          entire functions block
// Likelihood rewritten to take precomputed intercept subscripts ia, ia2,
// sgn (one triple per observation) as data instead of the raw 2-column
// censored y.  These are computed in R by rms::ormIdx() -- the same
// Fortran routine rms::orm.fit() uses -- from y, y2, and k, so the Stan
// code itself no longer branches on censoring type at all: the earlier
// uncensored/left/right/interval-censored special cases (and the
// left-censoring pposcore[a+1] bug flagged in a prior version of this
// file -- it froze the non-PO score at pposcore[2] regardless of the
// censoring bound) are replaced by one formula applied uniformly.  See
// rms::ormIdx's documentation for the exact meaning of ia/ia2/sgn: the
// likelihood contribution is F(alpha[ia]+eta) [sgn=1] or
// 1-F(alpha[ia]+eta) [sgn=-1] when ia2==0, or
// F(alpha[ia]+eta) - F(alpha[ia2]+eta) when ia2>0 (sgn is always 1 in
// that case). The non-PO score for a component at alpha index m is
// pposcore[m+1], matching the existing (already-correct) back-scaling
// convention elsewhere in blrm.r (e.g. `d$pposcore[-1]`).
functions {
  // pointwise log-likelihood contributions
  vector pw_log_lik(vector alpha, vector beta, vector tau, vector pposcore,
	                vector gamma, array[] row_vector X, array[] row_vector Z,
					array[] int ia, array[] int ia2, vector sgn,
					array[] int cluster) {
    int N = size(X);
    vector[N] out;
	int q  = size(Z) > 0 ? cols(Z[1]) : 0;
	int Nc = num_elements(cluster);
    for (n in 1:N) {
    	real eta  = X[n] * beta;
		if(Nc > 0) eta = eta + gamma[cluster[n]];
		real zeta = q > 0 ? Z[n] * tau : 0.;
		real e1 = alpha[ia[n]] + eta + (q > 0 ? pposcore[ia[n] + 1] * zeta : 0.);
		if(ia2[n] > 0) {
			real e2 = alpha[ia2[n]] + eta + (q > 0 ? pposcore[ia2[n] + 1] * zeta : 0.);
			out[n] = log_inv_logit_diff(e1, e2);
		} else
			out[n] = log_inv_logit(sgn[n] * e1);
    }
    return out;
  }
}

data {
  int<lower = 1> N;   // number of observations
  int<lower = 1> p;   // number of predictors
	int<lower = 0> q;   // number of non-PO predictors in Z
  int<lower = 2> k;   // number of outcome categories
	int<lower = 0> cn;  // number of contrasts needing priors
	int<lower = 0> cn2; // number of non-PO contrasts needing priors
	int<lower = 0, upper = k> lpposcore;  // extent of pposcore (1=PO)
  matrix[N, p] X;     // matrix of CENTERED predictors
	matrix[N, q] Z;     // matrix of CENTERED PPO predictors
	matrix[cn, p] C;    // contrasts
	matrix[cn2, q] C2;  // non-PO contrasts
  // Claude Sonnet 5 2026-11-14          3 lines
  // Precomputed via rms::ormIdx() in R; see functions block comment.
  array[N] int<lower = 1, upper = k - 1> ia;  // intercept index for p1
  array[N] int<lower = 0, upper = k - 1> ia2; // intercept index for p2 (0 = none)
  vector[N] sgn;                              // +1 if p1 is F(), -1 if 1-F()
	vector[lpposcore] pposcore; // scores for constrained partial PO
	int<lower = 0> Nc;  // number of clusters (0=no clustering)
	array[Nc == 0 ? 0 : N] int<lower = 1, upper = Nc> cluster;

  real<lower = 0> conc;

// priors for contrasts
	 vector[cn] cmus;
	 vector[cn] csds;
	 vector[cn2] cmus2;
	 vector[cn2] csds2;

  int<lower = 1, upper = 2> psigma;  // 1=t(4, rsdmean[1], rsdsd[1]), 2=exponential
	array[Nc == 0 ? 0 : 1] real<lower = 0> rsdmean;
	array[Nc == 0 || psigma == 2 ? 0 : 1] real<lower = 0> rsdsd;
}

transformed data {
	array[N] row_vector[p] Xr;
	array[N] row_vector[q] Zr;
  for (n in 1:N) Xr[n] = X[n, ];
	for (n in 1:N) Zr[n] = Z[n, ];
}

parameters {
  vector[p] beta; // coefficients on X
  vector[q] tau;  // coefficients on Z
  simplex[k] pi;  // category probabilities for a person w/ average predictors
	vector[Nc] gamma_raw;   // unscaled random effects
	array[Nc == 0 ? 0 : 1] real<lower = 0> sigmag; // SD of random effects
}

transformed parameters {
  vector[k - 1] alpha;                               // intercepts
  vector[Nc] gamma = Nc == 0 ? gamma_raw : sigmag[1] * gamma_raw;
  vector[N] log_lik;                                 // log-likelihood pieces
  for (j in 2:k) alpha[j - 1] = logit(sum(pi[j:k])); // predictors are CENTERED
  log_lik = pw_log_lik(alpha, beta, tau, pposcore, gamma, Xr, Zr, ia, ia2, sgn, cluster);
}

model {
  if(Nc > 0) {
    gamma_raw ~ std_normal();  // implies gamma ~ normal(0, sigmag)
	  if(psigma == 1) sigmag ~ student_t(4, rsdmean[1], rsdsd[1]);
		else sigmag ~ exponential(1. / rsdmean[1]);
		}
  target += log_lik;
  target += dirichlet_lpdf(pi | rep_vector(conc, k));
	if(cn > 0)  target += normal_lpdf(C * beta | cmus, csds);
	if(cn2 > 0) target += normal_lpdf(C2 * tau | cmus2, csds2);
}
