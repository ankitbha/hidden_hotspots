// STHM spatio-temp hierarchical model | v0.7.2
// Meas eq:
//   y_k(s,t) = g_k(eta_k(s,t)) + eps_k(s,t),
// where the linear predictor eta is
//   eta_k(s,t) = Z_k(s,t)*beta_k + B1(t)*alpha1 + B2(t)*alpha2 + X(s,t)
// and the link function gk is approximated by a quadratic polynomial:
//   g_k(x) = gamma_{1,k}*x + gamma_{2,k}*x^2.
// Proc eq:
//   X(s,t) = phi*X(s,t-1) + delta(s,t).
// The terms are:
//   - the response values y_k(s,t) are stacked in obsvec vector
//   - k=1,2 identifies two types of locations with distinct link function g_k
//     to be approximated by a quad polynomial, distinct effects in Z*beta
//     (in particular distinct intercepts), and distinct meas err sd sigmaeps
//   - Z_k(s,t) is p-vector of fixed covariates, incl intercept so p>=1
//   - B1(t) is a J1-vector of quad B-spline bases for daily seas effect, res 1
//   - B2(t) is a J2-vector of quad B-spline bases for weekly seas effect, res 2
//   - X(s,t) is dynamic spatial random field, GP (GRF) with mean 0, common to
//     two location types (k=1,2)
//   - eps(s,t) is iid Gaussian, mean 0, variance sigmaeps^2_k only varying by
//     location type
//   - delta(s,t) is Gaussian, mean 0, temporally indep, st isotropic expo cov
//     in space with microscale sd sigmadelta and expo decay slope gamma
//   - X(s,0) is from st dist N(0,sigmadelta^2/(1-phi^2)).
// Convention for vector stacking: space = outer loop, time = inner loop
// Missing values: obsind = 0-1 indicator of missing value in y, same length and
//                 ordering as obsvec. X is predicted everywhere and at all time
//                 points, so that obsind can be used to restrict obs likelihood
//                 contribution to sampled locations/times.
// Dimensions: nS = total nb of locations, nT = total nb of time points
// Change log from previous versions:
//   - quad poly only for approx link
//   - adjust meas mean eq to account for 2 location types: now quad polynomial
//     is for entire linear predictor, not just for Xvec. More meaningful.
//   - 2 different locations types, Xvec enters mean eq for obsvec as a quad
//     polynomial with different coef
//   - add a 2nd level of resolution for seas, res 1 = daily and res 2 = weekly
//   - enforce mean zero constraint for B*alpha so intercept now part of z*beta


#include <TMB.hpp>

// library needed for the multivariate normal distribution
using namespace density;

// simpler than pow(,2)
template <class Type> 
Type sqr(Type x){
	return x*x;
}

// // simpler than pow(,3)
// template <class Type> 
// Type cub(Type x){
// 	return x*x*x;
// }

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
	DATA_VECTOR(obsvec); // response variable log PM measurements, dim nT*nS
	// ^ incl NA entries from padding inR, obsind sorts them
	DATA_IVECTOR(obsind); // 1 if obsvec available, 0 if missing, dim nT*nS
	DATA_IVECTOR(loctype); // 1 if loc type 1, 2 if loc type 2, dim nT*nS
	DATA_MATRIX(Zmat1); // matrix of det covariates, loc type 1, dim (nT*nS x p)
	DATA_MATRIX(Zmat2); // matrix of det covariates, loc type 2, dim (nT*nS x p)
	// ^ at least one col of ones for intercept in beta
	DATA_MATRIX(Bmat1); // B-splines bases res 1, dim (nT*nS x J1)
	DATA_MATRIX(Bmat2); // B-splines bases res 2, dim (nT*nS x J2)
	DATA_VECTOR(kn1); // B-spline knots res 1, incl boundaries, dim J1-1
	DATA_VECTOR(kn2); // B-spline knots res 2, incl boundaries, dim J2-1
	DATA_MATRIX(distmat); // distances between locations, dim (nS x nS)


	// Fixed parameters
	PARAMETER_VECTOR(beta1); // det fx, incl intercept, loc type 1, dim p
	PARAMETER_VECTOR(beta2); // det fx, incl intercept, loc type 2, dim p
	PARAMETER_VECTOR(alpha1); // daily seas (res 1), B-spline coeff, dim J1-3
	PARAMETER_VECTOR(alpha2); // weekly seas (res 2), B-spline coeff, dim J2-3
	PARAMETER_VECTOR(gamma1); // coef quad poly approx link, loc type 1, dim 2
	PARAMETER_VECTOR(gamma2); // coef quad poly approx link, loc type 2, dim 2
	PARAMETER(log_sigmaeps1); // log sd meas error, loc type 1
	PARAMETER(log_sigmaeps2); // log sd meas error, loc type 2
	PARAMETER(t_phi); // transformed AR(1) coeff
	PARAMETER(log_gamma); // log expo decaying spatial covariance
	PARAMETER(log_sigmadelta); // log sd spatial Gaussian proc for ||h||=0
	
	// Random effects
	PARAMETER_VECTOR(X); // AR(1) in time + spatially corr proc err, dim nT*nS

	// misc
	// DATA_INTEGER(interceptonly); // 0=z(s,t)*beta+season, 1=only intercept
	// DATA_INTEGER(codedetfx); // 0=intercept only, 1=cov+seas, 2=seas only


	//--------------------------------------------------------------------------
	// Setup, procedures and init
	//--------------------------------------------------------------------------

	int i,j,t; // init counters

	int n = obsvec.size(); // n = nT*nS = total number of obs
	int nS = distmat.rows(); // total number of locations, incl missing values
	int nT = n/nS; // total number of time points, incl missing values
	int J1 = Bmat1.cols(); // nb quad B-spline bases
	int J2 = Bmat2.cols(); // nb quad B-spline bases
	// nb bases = J, nb knots = J-1, nb intervals = J-2, nb free alpha = J-3

	Type sigmaeps1 = exp(log_sigmaeps1); // loc type 1
	Type sigmaeps2 = exp(log_sigmaeps2); // loc type 2
	Type phi = bound11(t_phi); // transform so that within [-1,+1]
	Type gamma = exp(log_gamma);
	Type sigmadelta = exp(log_sigmadelta); // needed for ADREPORT
	Type sigmadelta2 = sqr(sigmadelta);

	vector<Type> alphavec1(J1);  // vector of all seas coeff, incl constr

	if (J1==5){ // min with 4 knots, J1=5 bases, 2 free alpha1's
		for (i=0; i<(J1-3); i++){
			alphavec1(i) = alpha1(i); // 1st J1-3 free, last 3 constrained
		}
		alphavec1(J1-3) = -(alpha1(0)*(kn1(1)-kn1(2)+kn1(J1-2)-kn1(J1-3)) +
			alpha1(1)*(kn1(2)-kn1(0)) +
			(alpha1(0)+(alpha1(0)-alpha1(1))*(kn1(J1-2)-kn1(J1-3))/(kn1(1)-kn1(0)))*
			(kn1(J1-2)-kn1(J1-4)))/(kn1(J1-2)-kn1(J1-5));
		// ^ constraint: mean zero
		alphavec1(J1-2) = alpha1(0) + (alpha1(0)-alpha1(1))*
			(kn1(J1-2)-kn1(J1-3))/(kn1(1)-kn1(0));
		// ^ constraint: differentiability at boundaries
		alphavec1(J1-1) = alpha1(0);
		// ^ constraint: continuity at boundary
	} else {
		for (i=0; i<(J1-3); i++){
			alphavec1(i) = alpha1(i); // 1st J1-3 free, last 3 constrained
		}
		Type tmp = 0.0;
		for (i=2; i<(J1-3); i++){
			tmp += alpha1(i)*(kn1(i+1)-kn1(i-2));
		}
		// ^ sum(alpha1[3:(J1-3)]*(kn1[4:(J1-2)]-kn1[1:(J1-5)]))
		alphavec1(J1-3) = -(alpha1(0)*(kn1(1)-kn1(2)+kn1(J1-2)-kn1(J1-3)) +
			alpha1(1)*(kn1(2)-kn1(0)) +
			(alpha1(0)+(alpha1(0)-alpha1(1))*(kn1(J1-2)-kn1(J1-3))/(kn1(1)-kn1(0)))*
			(kn1(J1-2)-kn1(J1-4)) + tmp)/(kn1(J1-2)-kn1(J1-5));
		// ^ constraint: mean zero
		alphavec1(J1-2) = alpha1(0) + (alpha1(0)-alpha1(1))*
			(kn1(J1-2)-kn1(J1-3))/(kn1(1)-kn1(0));
		// ^ constraint: differentiability at boundaries
		alphavec1(J1-1) = alpha1(0);
		// ^ constraint: continuity at boundary
	}

	vector<Type> alphavec2(J2);  // vector of all seas coeff, incl constr

	if (J2==5){ // min with 4 knots, J2=5 bases, 2 free alpha2's
		for (i=0; i<(J2-3); i++){
			alphavec2(i) = alpha2(i); // 1st J2-3 free, last 3 constrained
		}
		alphavec2(J2-3) = -(alpha2(0)*(kn2(1)-kn2(2)+kn2(J2-2)-kn2(J2-3)) +
			alpha2(1)*(kn2(2)-kn2(0)) +
			(alpha2(0)+(alpha2(0)-alpha2(1))*(kn2(J2-2)-kn2(J2-3))/(kn2(1)-kn2(0)))*
			(kn2(J2-2)-kn2(J2-4)))/(kn2(J2-2)-kn2(J2-5));
		// ^ constraint: mean zero
		alphavec2(J2-2) = alpha2(0) + (alpha2(0)-alpha2(1))*
			(kn2(J2-2)-kn2(J2-3))/(kn2(1)-kn2(0));
		// ^ constraint: differentiability at boundaries
		alphavec2(J2-1) = alpha2(0);
		// ^ constraint: continuity at boundary
	} else {
		for (i=0; i<(J2-3); i++){
			alphavec2(i) = alpha2(i); // 1st J2-3 free, last 3 constrained
		}
		Type tmp = 0.0;
		for (i=2; i<(J2-3); i++){
			tmp += alpha2(i)*(kn2(i+1)-kn2(i-2));
		}
		// ^ sum(alpha2[3:(J2-3)]*(kn2[4:(J2-2)]-kn2[1:(J2-5)]))
		alphavec2(J2-3) = -(alpha2(0)*(kn2(1)-kn2(2)+kn2(J2-2)-kn2(J2-3)) +
			alpha2(1)*(kn2(2)-kn2(0)) +
			(alpha2(0)+(alpha2(0)-alpha2(1))*(kn2(J2-2)-kn2(J2-3))/(kn2(1)-kn2(0)))*
			(kn2(J2-2)-kn2(J2-4)) + tmp)/(kn2(J2-2)-kn2(J2-5));
		// ^ constraint: mean zero
		alphavec2(J2-2) = alpha2(0) + (alpha2(0)-alpha2(1))*
			(kn2(J2-2)-kn2(J2-3))/(kn2(1)-kn2(0));
		// ^ constraint: differentiability at boundaries
		alphavec2(J2-1) = alpha2(0);
		// ^ constraint: continuity at boundary
	}

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

	matrix<Type> covdelta0 = covdelta/(Type(1.0)-sqr(phi)); // st cov
	MVNORM_t<Type> neglogdens_delta0(covdelta0); // multinorm for st dist, ini

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

	vector<Type> zb1 = Zmat1*beta1; // lin comb, loc type 1, dim nT*nS=n
	vector<Type> zb2 = Zmat2*beta2; // lin comb, loc type 2, dim nT*nS=n

	vector<Type> seas1 = Bmat1*alphavec1; // lin comb, dim nT*nS=n
	vector<Type> seas2 = Bmat2*alphavec2; // lin comb, dim nT*nS=n

	vector<Type> detfx(n); // fixed effects in lin pred
	vector<Type> linpred(n); // linear predictor = detfx + X
	vector<Type> fitted(n); // fitted values, quad poly in lin pred

	for (i = 0; i < n; i++){ // loop over all observations
		if (loctype(i)==1){
			detfx(i) = zb1(i) + seas1(i) + seas2(i);
			linpred(i) = detfx(i) + X(i);
			fitted(i) = gamma1(0)*linpred(i) + gamma1(1)*sqr(linpred(i));
			// + gamma1(2)*cub(linpred(i));
			if (obsind(i)==1){ // lkhd contrib only where/when data available
				nll -= dnorm(obsvec(i), fitted(i), sigmaeps1, true);
			}
		} else {
			detfx(i) = zb2(i) + seas1(i) + seas2(i);
			linpred(i) = detfx(i) + X(i);
			fitted(i) = gamma2(0)*linpred(i) + gamma2(1)*sqr(linpred(i));
			// + gamma2(2)*cub(linpred(i));
			if (obsind(i)==1){ // lkhd contrib only where/when data available
				nll -= dnorm(obsvec(i), fitted(i), sigmaeps2, true);
			}
		}
	}


	//--------------------------------------------------------------------------
	// Outputs
	//--------------------------------------------------------------------------
  
	// ADREPORT(beta); // necessary?
	// ADREPORT(alphavec);
	// ADREPORT(sigmaeps);
	// ADREPORT(phi);
	// ADREPORT(gamma);
	// ADREPORT(sigmadelta);

	REPORT(beta1);
	REPORT(beta2);
	REPORT(alphavec1);
	REPORT(alphavec2);
	REPORT(gamma1);
	REPORT(gamma2);
	REPORT(sigmaeps1);
	REPORT(sigmaeps2);
	REPORT(phi);
	REPORT(gamma);
	REPORT(sigmadelta);

	REPORT(zb1);
	REPORT(zb2);
	REPORT(seas1);
	REPORT(seas2);
	REPORT(detfx); // Z*beta + seas1 + seas2
	REPORT(linpred); // detfx + randeff
	REPORT(fitted); // quad poly in lin pred, distinct coeff by loc type

	// ADREPORT(fitted); // non-linear transfor of randef, need delta method
	// ^ warning: increases compute time for sdreport substantially

	return nll;
}
