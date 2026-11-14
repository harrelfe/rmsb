// For constrained partial PO model with t distribution for priors on intercepts

// Claude Sonnet 5 2026-11-14          entire functions block
// Likelihood rewritten to take precomputed intercept subscripts ia, ia2,
// sgn as data instead of the raw 2-column censored y; see lrmconppo.stan
// for the full rationale. alpha here is negated throughout (unchanged
// from before) because it is declared as a Stan `ordered` vector.
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
			real e1 = - alpha[ia[n]] + eta + (q > 0 ? pposcore[ia[n] + 1] * zeta : 0.);
			if(ia2[n] > 0) {
				real e2 = - alpha[ia2[n]] + eta + (q > 0 ? pposcore[ia2[n] + 1] * zeta : 0.);
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
	int<lower = 0> cn;  // number of contrasts (rows of C)
	int<lower = 0> cn2; // number of contrasts for non-PO (rows of C2)
	int<lower = 0, upper = k> lpposcore;  // extent of pposcore (1=PO)
  matrix[N, p] X;     // matrix of CENTERED predictors
	matrix[N, q] Z;     // matrix of CENTERED PPO predictors
	matrix[cn, p] C;    // contrasts
	matrix[cn2, q] C2;  // non-PO contrasts corresponding to Z
  // Claude Sonnet 5 2026-11-14          3 lines
  // Precomputed via rms::ormIdx() in R; see lrmconppo.stan.
  array[N] int<lower = 1, upper = k - 1> ia;  // intercept index for p1
  array[N] int<lower = 0, upper = k - 1> ia2; // intercept index for p2 (0 = none)
  vector[N] sgn;                              // +1 if p1 is F(), -1 if 1-F()
	vector[lpposcore] pposcore; // scores for constrained partial PO
	int<lower = 0> Nc;  // number of clusters (0=no clustering)
	array[Nc == 0 ? 0 : N] int<lower = 1, upper = Nc> cluster;

	int<lower = 1, upper = 2> iprior; // 1=flat 2=t(3, 0, ascale)
	real<lower = 0.01> ascale;

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
  ordered[k - 1] alpha;
  vector[p] beta; // coefficients on X
  vector[q] tau;  // coefficients on Z
	vector[Nc] gamma_raw;   // unscaled random effects
	array[Nc == 0 ? 0 : 1] real<lower = 0> sigmag; // SD of random effects
}

transformed parameters {
  vector[Nc] gamma = Nc == 0 ? gamma_raw : sigmag[1] * gamma_raw;
  vector[N] log_lik;                                 // log-likelihood pieces
  log_lik = pw_log_lik(alpha, beta, tau, pposcore, gamma, Xr, Zr, ia, ia2, sgn, cluster);
}

model {
  if(Nc > 0) {
    gamma_raw ~ std_normal();  // implies gamma ~ normal(0, sigmag)
	  if(psigma == 1) sigmag ~ student_t(4, rsdmean[1], rsdsd[1]);
		else sigmag ~ exponential(1. / rsdmean[1]);
		}
  target += log_lik;
	if(iprior == 2) target += student_t_lpdf(alpha | 3, 0., ascale);
	if(cn > 0)  target += normal_lpdf(C * beta | cmus, csds);
	if(cn2 > 0) target += normal_lpdf(C2 * tau | cmus2, csds2);
}
