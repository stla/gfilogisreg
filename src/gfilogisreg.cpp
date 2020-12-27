#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/special_functions/sin_pi.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "RcppArmadillo.h"
#include "roptim.h"
using namespace roptim;
namespace mp = boost::multiprecision;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]
// [[Rcpp::depends(BH)]]

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);

const double pi = boost::math::constants::pi<double>();

const double Epsilon = pow(std::numeric_limits<double>::epsilon(), 0.5);

const double Factor = 1.0e8;

double powint(double base, size_t exp) {
  double result = 1.0;
  while(exp) {
    if(exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

arma::vec tan01(const arma::vec& u) {
  return arma::log(u / (1.0 - u));  //arma::tan(pi * (u - 0.5)); //
}

double tan01scalar(double u) {
  return log(u / (1.0 - u)); //tan(pi * (u - 0.5)); //
}

double atan01(double x) {
  return 1.0 / (1.0 + exp(-x));//atan(x) / pi + 0.5;//
}

double dtan01(double u) {
//  const double x = boost::math::cos_pi(u - 0.5);
//  const double x = cos(pi*(u - 0.5));
//  return pi / (x * x);
  return 1.0 / (u * (1.0-u));
}

arma::vec dlogis(const arma::vec& x) {
  const arma::vec expminusx = arma::exp(-x);
  const arma::vec one_plus_expminusx = 1.0 + expminusx;
  return expminusx /
         (one_plus_expminusx % one_plus_expminusx);  // (1.0 / one_plus_expx) % (1.0 - 1.0
                                           // / one_plus_expx);
}

arma::vec ldlogis(const arma::vec& x) {
  return -x - 2.0 * arma::log1p(arma::exp(-x));
}
arma::vec dldlogis(const arma::vec& x) {
  return 1.0 - 2.0 / (1.0 + arma::exp(-x));
}

double forig(const arma::vec& x, const arma::mat& P, const arma::vec& b) {
  return arma::prod(dlogis(P * x + b));
}

double f(const arma::vec& u, const arma::mat& P, const arma::vec& b) {
  double result = arma::prod(dlogis(P * tan01(u) + b));
  return isnormal(result) ? result : 0.0;
//  return arma::prod(dlogis(P * tan01(u) + b));
}

double logf(const arma::vec& u, const arma::mat& P, const arma::vec& b) {
  return arma::sum(ldlogis(P * tan01(u) + b));
}

double df(const double ui,
          const arma::vec& Pi,
          const double y1,
          const arma::vec& y2) {
//  return y1 * dtan01(ui) * arma::sum(Pi % y2);
  double result = y1 * dtan01(ui) * arma::sum(Pi % y2);
  return isnormal(result) ? result : 0.0;
}

double dlogf(const double ui,
          const arma::vec& Pi,
          const arma::vec& y2) {
  return dtan01(ui) * arma::sum(Pi % y2);
}

class F : public Functor {
 public:
  arma::mat P;
  arma::vec b;
  double operator()(const arma::vec& u) override { return logf(u, P, b); }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y2 = dldlogis(P * tan01(u) + b);
    for(size_t i = 0; i < d; i++) {
      gr(i) = dlogf(u.at(i), P.col(i), y2);
    }
  }
};

// [[Rcpp::export]]
void xf2(arma::vec& u, arma::mat& P, arma::vec& b, arma::vec& mu, size_t j){
  const size_t d = P.n_cols;
  const double x = tan01scalar(u.at(j));
  const double y1 = f(u, P, b);
  Rcpp::Rcout << x << "\n";
  Rcpp::Rcout << y1 << "\n";
  Rcpp::Rcout << pow(y1, 1.0/(d+2)) << "\n";
  Rcpp::Rcout << 0.0 + pow(f(u, P, b), 1.0/(d+2)) * (tan01scalar(u.at(j)) - mu.at(j)) << "\n";
}

class xF : public Functor {
 public:
  arma::mat P;
  arma::vec b;
  arma::vec mu;
  size_t j;
  double operator()(const arma::vec& u) override {
    const size_t d = P.n_cols;
//    return pow(f(u, P, b), 1.0 / (d + 2)) * (tan01scalar(u.at(j)) - mu.at(j));
    const double result = pow(f(u, P, b), 1.0 / (d + 2)) * (tan01scalar(u.at(j)) - mu.at(j));
    return isnormal(result) ? result : 0.0;
  }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    const double alpha = 1.0 / (d + 2);
    gr = arma::zeros<arma::vec>(d);
    const double y1alpha = pow(f(u, P, b), alpha);
    const arma::vec y2 = dldlogis(P * tan01(u) + b);
    // alpha * prod(dlogis(vecx))^alpha * dh(uv[i]) * sum(P[, i] *
    // dldlogis(vecx))
    const double diff = tan01scalar(u.at(j)) - mu.at(j);
    for(size_t i = 0; i < d; i++) {
      const double z = y1alpha * dtan01(u.at(i));
//      const double dfalpha =
//          alpha * z * arma::sum(P.col(i) % y2);
      double result;
      if(i == j) {
        result = z * (alpha * arma::sum(P.col(i) % y2) * diff + 1.0);
      } else {
        result = alpha * z * arma::sum(P.col(i) % y2) * diff;
      }
      gr(i) = isnormal(result) ? result : 0.0;
    }
  }
};

// [[Rcpp::export]]
Rcpp::List get_umax(const arma::mat& P, const arma::vec& b, arma::vec init) {
  F optimand;
  optimand.P = P;
  optimand.b = b;
  const size_t d = P.n_cols;
  Roptim<F> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  opt.control.fnscale = -1.0;  // maximize
  opt.control.factr = Factor;
//  opt.control.pgtol = 1.0e-10;
  opt.control.lmm = 100;
  opt.set_hessian(false);
  arma::vec lwr = arma::zeros(d) + Epsilon;
  arma::vec upr = arma::ones(d) - Epsilon;
  opt.set_lower(lwr);
  opt.set_upper(upr);
  opt.minimize(optimand, init);
  if(opt.convergence() != 0) {
    Rcpp::Rcout << "-- umax -----------------------" << std::endl;
    opt.print();
  }
  return Rcpp::List::create(
      Rcpp::Named("mu") = tan01(opt.par()),
      Rcpp::Named("umax") = pow(exp(opt.value()), 2.0 / (d + 2)));
}

// [[Rcpp::export]]
double get_vmin_i(const arma::mat& P,
                  const arma::vec& b,
                  const size_t i,
                  const arma::vec& mu) {
  xF optimand;
  optimand.P = P;
  optimand.b = b;
  optimand.j = i;
  optimand.mu = mu;
  Roptim<xF> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  // opt.control.fnscale = 1.0;  // minimize
  opt.control.factr = 1.0e6;
  opt.control.lmm = 100;
  opt.set_hessian(false);
  const size_t d = P.n_cols;
  arma::vec init = 0.5 * arma::ones(d);
  init.at(i) = atan01(mu.at(i)) / 2.0;
  arma::vec lwr = arma::zeros(d) + Epsilon;
  arma::vec upr = arma::ones(d) - Epsilon;
  upr.at(i) = atan01(mu.at(i));
  opt.set_lower(lwr);
  opt.set_upper(upr);
  opt.minimize(optimand, init);
  if(opt.convergence() != 0) {
    Rcpp::Rcout << "-- vmin -----------------------" << std::endl;
    opt.print();
  }
  // Rcpp::Rcout << "-------------------------" << std::endl;
  return opt.value();
}

// [[Rcpp::export]]
arma::vec get_vmin(const arma::mat& P,
                   const arma::vec& b,
                   const arma::vec& mu) {
  const size_t d = P.n_cols;
  arma::vec vmin(d);
  for(size_t i = 0; i < d; i++) {
    vmin.at(i) = get_vmin_i(P, b, i, mu);
  }
  return vmin;
}

double get_vmax_i(const arma::mat& P,
                  const arma::vec& b,
                  const size_t i,
                  const arma::vec& mu) {
  xF optimand;
  optimand.P = P;
  optimand.b = b;
  optimand.j = i;
  optimand.mu = mu;
  Roptim<xF> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  opt.control.fnscale = -1.0;  // maximize
  opt.control.factr = 1.0e6;
  opt.control.lmm = 100;
  opt.set_hessian(false);
  const size_t d = P.n_cols;
  arma::vec init = 0.5 * arma::ones(d);
  init.at(i) = (atan01(mu.at(i)) + 1.0) / 2.0;
  arma::vec lwr = arma::zeros(d) + Epsilon;
  lwr.at(i) = atan01(mu.at(i));
  arma::vec upr = arma::ones(d) - Epsilon;
  opt.set_lower(lwr);
  opt.set_upper(upr);
  opt.minimize(optimand, init);
  if(opt.convergence() != 0) {
    Rcpp::Rcout << "-- vmax -----------------------" << std::endl;
    opt.print();
  }
  return opt.value();
}

// [[Rcpp::export]]
arma::vec get_vmax(const arma::mat& P,
                   const arma::vec& b,
                   const arma::vec& mu) {
  const size_t d = P.n_cols;
  arma::vec vmax(d);
  for(size_t i = 0; i < d; i++) {
    vmax.at(i) = get_vmax_i(P, b, i, mu);
  }
  return vmax;
}

// [[Rcpp::export]]
Rcpp::List get_bounds(const arma::mat& P, const arma::vec& b, arma::vec init) {
  Rcpp::List L = get_umax(P, b, init);
  arma::vec mu = L["mu"];
  double umax = L["umax"];
  arma::vec vmin = get_vmin(P, b, mu);
  arma::vec vmax = get_vmax(P, b, mu);
  return Rcpp::List::create(Rcpp::Named("umax") = umax, Rcpp::Named("mu") = mu,
                            Rcpp::Named("vmin") = vmin,
                            Rcpp::Named("vmax") = vmax);
}

// std::uniform_real_distribution<double> runif(0.0, 1.0);
// std::default_random_engine generator(seed);
// runif(generator)

// [[Rcpp::export]]
arma::mat rcd(const size_t n, const arma::mat& P, const arma::vec& b, arma::vec init) {
  std::uniform_real_distribution<double> runif(0.0, 1.0);
  const size_t d = P.n_cols;
  arma::mat tout(d, n);
  const Rcpp::List bounds = get_bounds(P, b, init);
  const double umax = bounds["umax"];
  const arma::vec mu = bounds["mu"];
  const arma::vec vmin = bounds["vmin"];
  const arma::vec vmax = bounds["vmax"];
  size_t k = 0;
  while(k < n) {
    const double u = umax * runif(generator);
    arma::vec v(d);
    for(size_t i = 0; i < d; i++) {
      v.at(i) = vmin.at(i) + (vmax.at(i) - vmin.at(i)) * runif(generator);
    }
    const arma::vec x = v / sqrt(u) + mu;
    if(u < pow(forig(x, P, b), 2.0/(d+2))) {
      tout.col(k) = x;
      k++;
    }
  }
  return tout.t();
}

////////////////////////////////////////////////////////////////////////////////
double plogis(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double qlogis(double u) {
  return log(u / (1.0 - u));
}

double MachineEps = std::numeric_limits<double>::epsilon();

double rtlogis1(double x, std::default_random_engine gen) {
  double b = plogis(x);
  if(b <= MachineEps) {
    Rcpp::Rcout << "b <= MachineEps\n";
    return x;
  }
  std::uniform_real_distribution<double> ru(MachineEps, b);
  return qlogis(ru(gen));
}

double rtlogis2(double x, std::default_random_engine gen) {
  double a = plogis(x);
  if(a == 1) {
    Rcpp::Rcout << "a==1\n";
    return x;
  }
  std::uniform_real_distribution<double> ru(a, 1);
  return qlogis(ru(gen));
}

std::string scalar2q(double x) {
  mp::mpq_rational q(x);
  return q.convert_to<std::string>();
}

Rcpp::CharacterVector vector2q(arma::colvec& x) {
  Rcpp::CharacterVector out(x.size());
  for(auto i = 0; i < x.size(); i++) {
    mp::mpq_rational q(x(i));
    out(i) = q.convert_to<std::string>();
  }
  return out;
}

Rcpp::CharacterVector newColumn(const arma::colvec& Xt,
                                double atilde,
                                const bool yzero) {
  arma::colvec head;
  arma::colvec newcol;
  if(yzero) {
    head = {0.0, atilde};
    newcol = arma::join_vert(head, -Xt);
  } else {
    head = {0.0, -atilde};
    newcol = arma::join_vert(head, Xt);
  }
  return vector2q(newcol);
}  // add column then transpose:

Rcpp::CharacterMatrix addHin(Rcpp::CharacterMatrix H,
                             const arma::colvec& Xt,
                             double atilde,
                             const bool yzero) {
  Rcpp::CharacterMatrix Ht = Rcpp::transpose(H);
  Rcpp::CharacterVector newcol = newColumn(Xt, atilde, yzero);
  Rcpp::CharacterMatrix Hnew = Rcpp::transpose(Rcpp::cbind(Ht, newcol));
  Hnew.attr("representation") = "H";
  return Hnew;
}

/*
Rcpp::List loop1(Rcpp::CharacterMatrix H,
                 const Rcpp::IntegerVector hbreaks,
                 const arma::mat& Points,
                 const Rcpp::IntegerVector pbreaks,
                 const int y,
                 const arma::colvec& Xt) {
  const size_t nthreads = 4;
  const size_t seed = 666;
  std::vector<std::default_random_engine> generators(nthreads);
  for(size_t t = 0; t < nthreads; t++) {
    std::default_random_engine gen(seed + (t + 1) * 2000000);
    generators[t] = gen;
  }
  const size_t N = hbreaks.size() - 1;
  const size_t p = H.cols() - 1;
  Rcpp::NumericVector weight(N);
  Rcpp::NumericVector At(N);
  Rcpp::List Hnew(N);
  if(y == 0) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(auto i = 0; i < N; i++) {
#ifdef _OPENMP
      const unsigned thread = omp_get_thread_num();
#else
      const unsigned thread = 0;
#endif
      arma::mat points = Points.rows(pbreaks(i), pbreaks(i + 1)-1);
      double MIN = arma::min(points * Xt);
      double atilde = rtlogis2(MIN, generators[thread]);
      At(i) = atilde;
      weight(i) = 1.0 - plogis(MIN);
#pragma omp critical
{
      Hnew[i] = addHin(H(Rcpp::Range(hbreaks(i), hbreaks(i + 1)-1),
Rcpp::Range(0, p)), Xt, atilde, true);
}
    }
  } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(auto i = 0; i < N; i++) {
#ifdef _OPENMP
      const unsigned thread = omp_get_thread_num();
#else
      const unsigned thread = 0;
#endif
      arma::mat points = Points.rows(pbreaks(i), pbreaks(i + 1)-1);
      double MAX = arma::max(points * Xt);
      double atilde = rtlogis1(MAX, generators[thread]);
      At(i) = atilde;
      weight(i) = plogis(MAX);
#pragma omp critical
{
      Hnew[i] = addHin(H(Rcpp::Range(hbreaks(i), hbreaks(i + 1)-1),
Rcpp::Range(0, p)), Xt, atilde, false);
}
    }
  }
  return Rcpp::List::create(Rcpp::Named("H") = Hnew, Rcpp::Named("At") = At,
                            Rcpp::Named("weight") = weight);
}
*/

// [[Rcpp::export]]
Rcpp::List loop1(Rcpp::List H,
                 const Rcpp::List Points,
                 const int y,
                 const arma::colvec& Xt) {
  // const size_t seed = 666;
  // std::vector<std::default_random_engine> generators(nthreads);
  // for(size_t t = 0; t < nthreads; t++) {
  //  std::default_random_engine gen(seed + (t + 1) * 2000000);
  //  generators[t] = gen;
  //}
  const size_t N = H.size();
  Rcpp::NumericVector weight(N);
  Rcpp::NumericVector At(N);
  if(y == 0) {
    for(auto i = 0; i < N; i++) {
      arma::mat points = Points[i];
      double MIN = arma::min(points * Xt);
      double atilde = rtlogis2(MIN, generator);
      At(i) = atilde;
      weight(i) = 1.0 - plogis(MIN);
      H[i] = addHin(H[i], Xt, atilde, true);
    }
  } else {
    for(auto i = 0; i < N; i++) {
      arma::mat points = Points[i];
      double MAX = arma::max(points * Xt);
      double atilde = rtlogis1(MAX, generator);
      At(i) = atilde;
      weight(i) = plogis(MAX);
      H[i] = addHin(H[i], Xt, atilde, false);
    }
  }
  return Rcpp::List::create(Rcpp::Named("H") = H, Rcpp::Named("At") = At,
                            Rcpp::Named("weight") = weight);
}
