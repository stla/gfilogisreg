#include <boost/multiprecision/gmp.hpp>
#include "RcppArmadillo.h"
#include "roptim.h"
using namespace roptim;
namespace mp = boost::multiprecision;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]
// [[Rcpp::depends(BH)]]

/*
std::vector<size_t> CantorExpansion(size_t n, std::vector<size_t> s) {
  std::vector<size_t> out(s.size());
  std::vector<size_t>::iterator it;
  it = s.begin();
  it = s.insert(it, 1);
  size_t G[s.size()];
  std::partial_sum(s.begin(), s.end(), G, std::multiplies<size_t>());
  size_t k;
  while(n > 0) {
    k = 1;
    while(G[k] <= n) {
      k++;
    }
    out[k - 1] = n / G[k - 1];
    n = n % G[k - 1];
  }
  return out;
}

arma::mat grid(const size_t d) {
  std::array<double, 3> x = {0.01, 0.5, 0.99};
  size_t p = pow((size_t)3, d);
  arma::mat out(d, p);
  std::vector<size_t> threes(d, 3);
  for(size_t n = 0; n < p; n++) {
    std::vector<size_t> indices = CantorExpansion(n, threes);
    for(size_t i = 0; i < d; i++) {
      out.at(i, n) = x[indices[i]];
    }
  }
  return out;
}
*/

arma::vec logit(const arma::vec& u) {
  return arma::log(u / (1.0 - u));
}

double dlogit(double u) {
  return 1.0 / (u * (1.0 - u));
}

arma::vec ldlogit(const arma::vec& u) {
  return -arma::log(u % (1.0 - u));
}

arma::vec ldlogis(const arma::vec& x) {
  return x - 2.0 * arma::log1p(arma::exp(x));
}

arma::vec dldlogis(const arma::vec& x) {
  return 1.0 - 2.0 / (1.0 + arma::exp(-x));
}

double log_f(const arma::vec& u, const arma::mat& P, const arma::vec& b) {
  const arma::vec x = P * logit(u) + b;
  return arma::sum(ldlogis(x)) + arma::sum(ldlogit(u));
}

double dlog_f(const double ui, const arma::vec& Pi, const arma::vec& y) {
  return dlogit(ui) * arma::sum(Pi % y) + (2.0 * ui - 1.0) / (ui * (1.0 - ui));
}

class Logf : public Functor {
 public:
  arma::mat P;
  arma::vec b;
  double operator()(const arma::vec& u) override { return log_f(u, P, b); }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y = dldlogis(P * logit(u) + b);
    for(size_t i = 0; i < d; i++) {
      gr(i) = dlog_f(u[i], P.col(i), y);
    }
  }
};

class uLogf1 : public Functor {
 public:
  arma::mat P;
  arma::vec b;
  arma::vec mu;
  size_t j;
  double operator()(const arma::vec& u) override {
    const size_t d = P.n_cols;
    return -log_f(u, P, b) - (d + 2) * log(mu[j] - u.at(j));
  }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y = dldlogis(P * logit(u) + b);
    for(size_t i = 0; i < d; i++) {
      if(i == j) {
        gr(i) = -dlog_f(u[i], P.col(i), y) + (d + 2) / (mu[i] - u.at(i));
      } else {
        gr(i) = -dlog_f(u[i], P.col(i), y);
      }
    }
  }
};

class uLogf2 : public Functor {
 public:
  arma::mat P;
  arma::vec b;
  arma::vec mu;
  size_t j;
  double operator()(const arma::vec& u) override {
    const size_t d = P.n_cols;
    return log_f(u, P, b) + (d + 2) * log(u.at(j) - mu[j]);
  }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y = dldlogis(P * logit(u) + b);
    for(size_t i = 0; i < d; i++) {
      if(i == j) {
        gr(i) = dlog_f(u[i], P.col(i), y) - (d + 2) / (mu[i] - u.at(i));
      } else {
        gr(i) = dlog_f(u[i], P.col(i), y);
      }
    }
  }
};

Rcpp::List get_umax0(const arma::mat& P, const arma::vec& b, arma::vec init) {
  double eps = sqrt(std::numeric_limits<double>::epsilon());
  Logf logf;
  logf.P = P;
  logf.b = b;
  Roptim<Logf> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  opt.control.fnscale = -1.0;  // maximize
  // opt.control.factr = 1.0;
  opt.set_hessian(false);
  arma::vec lwr = arma::zeros(init.size()) + eps;
  arma::vec upr = arma::ones(init.size()) - eps;
  opt.set_lower(lwr);
  opt.set_upper(upr);
  opt.minimize(logf, init);
  if(opt.convergence() != 0) {
    Rcpp::Rcout << "-- umax -----------------------" << std::endl;
    opt.print();
  }
  // Rcpp::Rcout << "-------------------------" << std::endl;
  //  opt.print();
  return Rcpp::List::create(Rcpp::Named("par") = opt.par(),
                            Rcpp::Named("value") = opt.value());
}

// [[Rcpp::export]]
Rcpp::List get_umax(const arma::mat& P,
                    const arma::vec& b,
                    const arma::mat& inits) {
  const size_t d = P.n_cols;
  // const arma::mat inits = grid(d);
  const size_t n = inits.n_cols;
  std::vector<arma::vec> pars(n);
  arma::vec values(n);
  for(size_t i = 0; i < n; i++) {
    const Rcpp::List L = get_umax0(P, b, inits.col(i));
    const arma::vec par = L["par"];
    pars[i] = par;
    // double value = L["value"];
    values(i) = L["value"];
  }
  const size_t imax = values.index_max();
  return Rcpp::List::create(
      Rcpp::Named("mu") = pars[imax],
      Rcpp::Named("umax") = pow(exp(values(imax)), 2.0 / (2.0 + d)));
}

// [[Rcpp::export]]
double get_vmin_i(const arma::mat& P,
                  const arma::vec& b,
                  const size_t i,
                  const arma::vec& mu) {
  double eps = sqrt(std::numeric_limits<double>::epsilon()) / 3.0;
  uLogf1 ulogf1;
  ulogf1.P = P;
  ulogf1.b = b;
  ulogf1.j = i;
  ulogf1.mu = mu;
  Roptim<uLogf1> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  // opt.control.fnscale = 1.0;  // minimize
  // opt.control.factr = 1.0;
  opt.set_hessian(false);
  const size_t d = P.n_cols;
  arma::vec init = 0.5 * arma::ones(d);
  init.at(i) = mu.at(i) / 2.0;
  arma::vec lwr = arma::zeros(d) + eps;
  arma::vec upr = arma::ones(d);
  upr.at(i) = mu.at(i);
  opt.set_lower(lwr);
  opt.set_upper(upr - eps);
  opt.minimize(ulogf1, init);
  if(opt.convergence() != 0) {
    Rcpp::Rcout << "-- vmin -----------------------" << std::endl;
    opt.print();
  }
  // Rcpp::Rcout << "-------------------------" << std::endl;
  return -exp(-opt.value() / (d + 2));
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
  double eps = sqrt(std::numeric_limits<double>::epsilon()) / 3.0;
  uLogf2 ulogf2;
  ulogf2.P = P;
  ulogf2.b = b;
  ulogf2.j = i;
  ulogf2.mu = mu;
  Roptim<uLogf2> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  opt.control.fnscale = -1.0;  // maximize
  // opt.control.factr = 1.0;
  opt.set_hessian(false);
  const size_t d = P.n_cols;
  arma::vec init = 0.5 * arma::ones(d);
  init.at(i) = (mu.at(i) + 1.0) / 2.0;
  arma::vec lwr = arma::zeros(d);
  lwr.at(i) = mu.at(i);
  arma::vec upr = arma::ones(d) - eps;
  opt.set_lower(lwr + eps);
  opt.set_upper(upr);
  opt.minimize(ulogf2, init);
  if(opt.convergence() != 0) {
    Rcpp::Rcout << "-- vmax -----------------------" << std::endl;
    opt.print();
  }
  return exp(opt.value() / (d + 2));
}

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
Rcpp::List get_bounds(const arma::mat& P,
                      const arma::vec& b,
                      const arma::mat& inits) {
  Rcpp::List L = get_umax(P, b, inits);
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
std::default_random_engine generator;
std::uniform_real_distribution<double> runif(0.0, 1.0);

// [[Rcpp::export]]
arma::mat rcd(const size_t n,
              const arma::mat& P,
              const arma::vec& b,
              const arma::mat& inits) {
  //, const size_t seed){
  //  std::default_random_engine generator(seed);
  //  std::uniform_real_distribution<double> runif(0.0, 1.0);
  const size_t d = P.n_cols;
  arma::mat tout(d, n);
  const Rcpp::List bounds = get_bounds(P, b, inits);
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
    bool test = arma::all(x > 0.0) && arma::all(x < 1.0) &&
                (d + 2) * log(u) < 2.0 * log_f(x, P, b);
    if(test) {
      tout.col(k) = logit(x);
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
    return x;
  }
  std::uniform_real_distribution<double> ru(MachineEps, b);
  return qlogis(ru(gen));
}

double rtlogis2(double x, std::default_random_engine gen) {
  double a = plogis(x);
  if(a == 1) {
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
  std::default_random_engine gnrtr;
  const size_t N = H.size();
  Rcpp::NumericVector weight(N);
  Rcpp::NumericVector At(N);
  if(y == 0) {
    for(auto i = 0; i < N; i++) {
      arma::mat points = Points[i];
      double MIN = arma::min(points * Xt);
      double atilde = rtlogis2(MIN, gnrtr);
      At(i) = atilde;
      weight(i) = 1.0 - plogis(MIN);
      H[i] = addHin(H[i], Xt, atilde, true);
    }
  } else {
    for(auto i = 0; i < N; i++) {
      arma::mat points = Points[i];
      double MAX = arma::max(points * Xt);
      double atilde = rtlogis1(MAX, gnrtr);
      At(i) = atilde;
      weight(i) = plogis(MAX);
      H[i] = addHin(H[i], Xt, atilde, false);
    }
  }
  return Rcpp::List::create(Rcpp::Named("H") = H, Rcpp::Named("At") = At,
                            Rcpp::Named("weight") = weight);
}
