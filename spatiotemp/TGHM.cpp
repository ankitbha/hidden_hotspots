// TGHM: toy geostatistical hierarchical model
//   meas eq: log y(s,t) = z(s,t)*beta + B(t)*alpha + X(s,t) + epsilon(s,t)
//   proc eq: X(s,t) = phi*X(s,t-1) + delta(s,t),
// where
//   - z(s,t) is p-vector of fixed covariate
//   - B(t) is a J-vector of quadratic B-spline bases for daily seasonal effect
//   - X(s,t) is dynamic spatial random field, mean 0
//   - epsilon(s,t) is iid Gaussian, mean 0, constant variance sigmaepsilon^2
//   - delta(s,t) is Gaussian, mean 0, temporally indep, st isotropic expo cov
//     in space with microscale sd sigmadelta and expo decay slope gamma
//   - X(s,0) is from st dist N(0,sigmadelta^2/(1-phi^2)).
// Convention for vector stacking: space = outer loop, time = inner loop
// Missing values: obsind = 0-1 indicator of missing value in y, same length and
//                 ordering as log_y. X is predicted everywhere and at all time
//                 points, so that obsind can be used to restrict obs likelihood
//                 contribution to sampled locations/times.
// Dimensions: nS = total nb of locations, nT = total nb of time points
// v0.5
//   - Added seasonal temporal effect with quadratic B-spline basis input as
//     fixed covariate. J=6 bases, 5 knots, 4 intervals partitioning one day:
//       * 00:00-05:45
//       * 06:00-11:45
//       * 12:00-17:45
//       * 18:00-23:45
//     J=6 alpha to estimate, only 4 free because constraints for continuity and
//     differentiability at boundaries. Intercept now absorbed in seasonal
//     effect.
//  - Now consider linear trend + weather covariates in zmat, no change below.

#include <TMB.hpp>

// library needed for the multivariate normal distribution
using namespace density;

// simpler than pow(,2)
template <class Type> 
Type square(Type x){
	return x*x;
}

// parameter transformation, R -> (-1,+1)
template <class Type>
Type bound11(Type x){
	return (Type(1.0)-exp(-x))/(Type(1.0)+exp(-x));
}

// objective function
template<class Type>

Type objective_function<Type>::operator() () {

	//--------------------------------------------------------------------------
	// Inputs
	//--------------------------------------------------------------------------

	// Data
	DATA_VECTOR(log_y); // response variable log PM measurements, dim nT*nS
	// ^ can include NA_real_ entries from R, obsind sorts them
	DATA_IVECTOR(obsind); // 1 if log_y available, 0 if missing, dim nT*nS
	DATA_MATRIX(zmat); // matrix of deterministic covariates, dim (nT*nS x p)
	DATA_MATRIX(Bmat); // matrix of B-splines bases, dim (nT*nS x J)
	DATA_VECTOR(kn); // vector of B-spline knots, incl boundaries, dim J-1
	DATA_MATRIX(distmat); // distances between locations, dim (nS x nS)

	// Fixed parameters
	PARAMETER_VECTOR(beta); // deterministic fixed fx, dim p
	PARAMETER_VECTOR(alpha); // daily seasonal effect, B-spline coeff, dim J-2
	PARAMETER(log_sigmaepsilon); // log sd meas error
	PARAMETER(t_phi); // transformed AR(1) coeff
	PARAMETER(log_gamma); // log expo decaying spatial covariance
	PARAMETER(log_sigmadelta); // log sd spatial Gaussian proc for ||h||=0
	
	// Random effects
	PARAMETER_VECTOR(X); // AR(1) in time + spatially corr proc err, dim nT*nS

	// misc
	DATA_INTEGER(interceptonly); // 0 = z(s,t)*beta+season , 1 = only intercept


	//--------------------------------------------------------------------------
	// Setup, procedures and init
	//--------------------------------------------------------------------------

	int i,j,t; // init counters

	int n = log_y.size(); // n = nT*nS = total number of obs
	int nS = distmat.rows(); // total number of locations, incl missing values
	int nT = n/nS; // total number of time points, incl missing values
	int J = alpha.size()+2; // nb quadratic splines for daily seasonality

	Type sigmaepsilon = exp(log_sigmaepsilon);
	Type phi = bound11(t_phi); // transform so that within [-1,+1]
	Type gamma = exp(log_gamma);
	Type sigmadelta = exp(log_sigmadelta); // needed for ADREPORT
	Type sigmadelta2 = square(sigmadelta);
	
	vector<Type> alphavec(J); // J=6 bases normally
	for (i=0; i<(J-2); i++){
		alphavec(i) = alpha(i); // 1st J-2 free, only last 2 constrained
	}
	alphavec(J-2) = alpha(0) + (alpha(0)-alpha(1))*
	                 (kn(J-2)-kn(J-3))/(kn(1)-kn(0));
	// ^ constraint: differentiability at boundaries
	alphavec(J-1) = alphavec(0); // constraint: continuity at boundaries

	Type nll = 0.0; // init neg loglik

	//--------------------------------------------------------------------------
	// Proc eq
	//--------------------------------------------------------------------------

	// delta: Gaussian field, mean 0, expo decaying covariance, temp indep
	matrix<Type> covdelta(nS,nS); // init cov mat of delta, nS locations
	for (i = 0; i < nS; i++){ // loop over locations
		covdelta(i,i) = sigmadelta2; // variance at ||h|| = 0
		for (j = 0; j < i; j++){
			covdelta(i,j) = sigmadelta2*exp(-distmat(i,j)/gamma); // expo decay
			covdelta(j,i) = covdelta(i,j); // symmetry
		}
	}
	MVNORM_t<Type> neglogdens_delta(covdelta); // multinorm for recursions

	matrix<Type> covdelta0 = covdelta/(Type(1.0)-square(phi)); // st cov
	MVNORM_t<Type> neglogdens_delta0(covdelta0); // multinorm for given covdelta

	// X: st dist at t=0
	vector<Type> diffX0(nS); // init vector of X - st mean
	for (i = 0; i < nS; i++){ // loop over locations
		int indst = nT*i; // maps indices, location i, time t=0
		// ^ convention: space = outer loop, time = inner loop
		diffX0(i) = X(indst); // mean 0
	}
	nll += neglogdens_delta0(diffX0);

	// X: recursions for t>=1
	for (t = 1; t < nT; t++){
		vector<Type> diffX(nS); // init vector of X - cond mean
		for (i = 0; i < nS; i++){ // loop over locations
			int indst = nT*i+t; // maps indices, location i, time t
			// ^ convention: space = outer loop, time = inner loop
			diffX(i) = X(indst) - phi*X(indst-1); // AR(1)
		}
		nll += neglogdens_delta(diffX);
	}

	//--------------------------------------------------------------------------
	// Obs eq
	//--------------------------------------------------------------------------

	if (interceptonly==0){ // covariates in zmat + daily seasonality
		vector<Type> detfx = zmat*beta; // lin comb, dim nT*nS=n
		vector<Type> season = Bmat*alphavec; // lin comb, dim nT*nS=n
		for (i = 0; i < n; i++){ // loop over all observations
			if (obsind(i)==1){ // lkhd contrib only where/when data available
				nll -= dnorm(log_y(i), detfx(i) + season(i) + X(i),
							 sigmaepsilon, true);
				// ^ iid measurement error
			}
		}
	} else { // intercept only, zmat ignored and only first entry of beta used
		Type detfx = beta(0); // intercept, constant across space-time
		for (i = 0; i < n; i++){ // loop over all observations
			if (obsind(i)==1){ // lkhd contrib only where/when data available
				nll -= dnorm(log_y(i), detfx + X(i), sigmaepsilon, true);
				// ^ iid measurement error
			}
		}
	}

	//--------------------------------------------------------------------------
	// Outputs
	//--------------------------------------------------------------------------
  
	// REPORT(covdelta);
	// REPORT(covdelta0);

	// REPORT(detfx);

	ADREPORT(beta); // necessary?
	ADREPORT(alphavec);
	ADREPORT(sigmaepsilon);
	ADREPORT(phi);
	ADREPORT(gamma);
	ADREPORT(sigmadelta);

	// ADREPORT(X); // not needed since not transformed

	return nll;
}
