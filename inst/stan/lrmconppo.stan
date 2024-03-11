// For constrained partial PO model with Dirichlet prior for transformed intercepts

functions {
  // pointwise log-likelihood contributions
  vector pw_log_lik(vector alpha, vector beta, vector tau, vector pposcore, 
	                vector gamma, array[] row_vector X, array[] row_vector Z, array[,] int y,
					array[] int cluster) {
    int N = size(X);
	real ll;
    vector[N] out;
    int k = max(max(y[,1]), max(y[,2])); // assumes all possible categories are observed
	real zeta = 0.;
	real r = 0.;
	real r2 = 0.;
	real ca;
  	real ca1;
	int a;
	int b;
	int q = size(Z) > 0 ? cols(Z[1]) : 0;
	int Nc = num_elements(cluster);
    for (n in 1:N) {
    	real eta  = X[n] * beta;
		if(Nc > 0) eta = eta + gamma[cluster[n]];
		if(q > 0)  zeta = Z[n] * tau;
    	a = y[n, 1];    // Y interval censored in [a,b] (uncensored if a=b)
		b = y[n, 2];
		if(a == 1 && b == k) ll = 0.;  // [min,max] interval censoring uninformative
		if(a == b) {    // uncensored
			if(q == 0) {  // PO
  				if (a == 1)        ca  = -(alpha[1]     + eta);
	  			else if (a == 2)   ca  =   alpha[1]     + eta;
		  	  	else               ca  =   alpha[a - 1] + eta;
			    if(a > 1 && a < k) ca1 =   alpha[a]     + eta;
			}
		else {
  			if (a == 1)        ca  = -( alpha[1]     + eta + pposcore[2]   * zeta);
	  		else if (a == 2)   ca  =    alpha[1]     + eta + pposcore[2]   * zeta;
		  	else               ca  =    alpha[a - 1] + eta + pposcore[a]   * zeta;
			if(a > 1 && a < k) ca1 =    alpha[a]     + eta + pposcore[a+1] * zeta;
			}
      	if (a == 1 || a == k) ll = log_inv_logit(ca);
		else ll = log(1./(1. + exp(-ca)) - 1./(1. + exp(-ca1)));
		}
		else if(b == k) {   // right censored at a
		if(q > 0) r = pposcore[a] * zeta;
		ll = log_inv_logit(alpha[a-1] + eta + r);
		}
		else if(a == 1) {   // left censored at b
			if(q > 0) r = pposcore[a+1] * zeta;
			ll = log_inv_logit(-(alpha[b] + eta + r));
			}
		else {              // a > a and b < k
			if(q > 0) {
				r  = pposcore[a]   * zeta;
				r2 = pposcore[b+1] * zeta;
				}
        ll = log(1./(1. + exp(-(alpha[a-1] + eta + r))) - 1./(1. + exp(-(alpha[b] + eta + r2)))); 
		}
			out[n] = ll;
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
  array[N, 2] int<lower = 1, upper = k> y; // 2-column outcome on 1 ... k
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
  log_lik = pw_log_lik(alpha, beta, tau, pposcore, gamma, Xr, Zr, y, cluster);
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
