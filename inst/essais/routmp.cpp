#include "RcppArmadillo.h"
#include "roptim.h"
using namespace roptim;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]

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

arma::vec logit(const arma::vec& u) {
  return arma::log(u / (1.0 - u));
}

double dlogit(double u) {
  return 1.0 / (u * (1.0 - u));
}

arma::vec ldlogit(const arma::vec& u) {
  return -arma::log(u % (1.0-u));
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
    return -log_f(u, P, b) - (d+2) * log(mu[j] - u.at(j));
  }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y = dldlogis(P * logit(u) + b);
    for(size_t i = 0; i < d; i++) {
      if(i == j){
        gr(i) = -dlog_f(u[i], P.col(i), y) + (d+2) / (mu[i] - u.at(i));
      }else{
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
    return log_f(u, P, b) + (d+2) * log(u.at(j) - mu[j]);
  }
  void Gradient(const arma::vec& u, arma::vec& gr) override {
    const size_t d = P.n_cols;
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y = dldlogis(P * logit(u) + b);
    for(size_t i = 0; i < d; i++) {
      if(i == j){
        gr(i) = dlog_f(u[i], P.col(i), y) - (d+2) / (mu[i] - u.at(i));
      }else{
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
  //opt.control.factr = 1.0;
  opt.set_hessian(false);
  arma::vec lwr = arma::zeros(init.size()) + eps;
  arma::vec upr = arma::ones(init.size()) - eps;
  opt.set_lower(lwr);
  opt.set_upper(upr);
  opt.minimize(logf, init);
  if(opt.convergence() != 0){
    Rcpp::Rcout << "-- umax -----------------------" << std::endl;
    opt.print();
  }
  //Rcpp::Rcout << "-------------------------" << std::endl;
  //  opt.print();
  return Rcpp::List::create(Rcpp::Named("par") = opt.par(),
                            Rcpp::Named("value") = opt.value());
}

// [[Rcpp::export]]
Rcpp::List get_umax(const arma::mat& P, const arma::vec& b) {
  const size_t d = P.n_cols;
  const arma::mat inits = grid(d);
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
double get_vmin_i(
    const arma::mat& P, const arma::vec& b, const size_t i, const arma::vec& mu
) {
  double eps = sqrt(std::numeric_limits<double>::epsilon()) / 3.0;
  uLogf1 ulogf1;
  ulogf1.P = P;
  ulogf1.b = b;
  ulogf1.j = i;
  ulogf1.mu = mu;
  Roptim<uLogf1> opt("L-BFGS-B");
  opt.control.trace = 0;
  opt.control.maxit = 10000;
  //opt.control.fnscale = 1.0;  // minimize
  //opt.control.factr = 1.0;
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
  if(opt.convergence() != 0){
    Rcpp::Rcout << "-- vmin -----------------------" << std::endl;
    opt.print();
  }
  //Rcpp::Rcout << "-------------------------" << std::endl;
  return -exp(-opt.value() / (d+2));
}

// [[Rcpp::export]]
arma::vec get_vmin(
    const arma::mat& P, const arma::vec& b, const arma::vec& mu
) {
  const size_t d = P.n_cols;
  arma::vec vmin(d);
  for(size_t i = 0; i < d; i++){
    vmin.at(i) = get_vmin_i(P, b, i, mu);
  }
  return vmin;
}

double get_vmax_i(
    const arma::mat& P, const arma::vec& b, const size_t i, const arma::vec& mu
) {
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
  //opt.control.factr = 1.0;
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
  if(opt.convergence() != 0){
    Rcpp::Rcout << "-- vmax -----------------------" << std::endl;
    opt.print();
  }
  return exp(opt.value() / (d+2));
}

arma::vec get_vmax(
    const arma::mat& P, const arma::vec& b, const arma::vec& mu
) {
  const size_t d = P.n_cols;
  arma::vec vmax(d);
  for(size_t i = 0; i < d; i++){
    vmax.at(i) = get_vmax_i(P, b, i, mu);
  }
  return vmax;
}

// [[Rcpp::export]]
Rcpp::List get_bounds(const arma::mat& P, const arma::vec& b){
  Rcpp::List L = get_umax(P, b);
  arma::vec mu = L["mu"];
  double umax = L["umax"];
  arma::vec vmin = get_vmin(P, b, mu);
  arma::vec vmax = get_vmax(P, b, mu);
  return Rcpp::List::create(Rcpp::Named("umax") = umax,
                            Rcpp::Named("mu") = mu,
                            Rcpp::Named("vmin") = vmin,
                            Rcpp::Named("vmax") = vmax);
}


// std::uniform_real_distribution<double> runif(0.0, 1.0);
// std::default_random_engine generator(seed);
// runif(generator)
std::default_random_engine generator;
std::uniform_real_distribution<double> runif(0.0, 1.0);

// [[Rcpp::export]]
arma::mat rcd(const size_t n, const arma::mat& P, const arma::vec& b){
  //, const size_t seed){
  //  std::default_random_engine generator(seed);
  //  std::uniform_real_distribution<double> runif(0.0, 1.0);
  const size_t d = P.n_cols;
  arma::mat tout(d, n);
  const Rcpp::List bounds = get_bounds(P, b);
  const double umax = bounds["umax"];
  const arma::vec mu = bounds["mu"];
  const arma::vec vmin = bounds["vmin"];
  const arma::vec vmax = bounds["vmax"];
  size_t k = 0;
  while(k < n){
    const double u = umax * runif(generator);
    arma::vec v(d);
    for(size_t i = 0; i < d; i++){
      v.at(i) = vmin.at(i) + (vmax.at(i) - vmin.at(i)) * runif(generator);
    }
    const arma::vec x = v / sqrt(u) + mu;
    bool test = arma::all(x > 0.0) && arma::all(x < 1.0) &&
      (d+2) * log(u) < 2.0 * log_f(x, P, b);
    if(test){
      tout.col(k) = logit(x);
      k++;
    }
  }
  return tout.t();
}


////////////////////////////////////////////////////////////////////////////////
double plogis(double x){
  return 1.0/(1.0 + exp(-x));
}

double qlogis(double u){
  return log(u/(1.0-u));
}

double MachineEps = std::numeric_limits<double>::epsilon();

double rlogis1(double x){
  double b = plogis(x);
  if(b <= MachineEps){
    return x;
  }
  std::uniform_real_distribution<double> ru(MachineEps, b);
  return qlogis(ru(generator));
}

double rlogis2(double x){
  double a = plogis(x);
  if(a == 1){
    return x;
  }
  std::uniform_real_distribution<double> ru(a, 1);
  return qlogis(ru(generator));
}









class Rosen : public Functor {
public:
  double h;
  double operator()(const arma::vec& x) override {
    double x1 = x(0);
    double x2 = x(1);
    return h * std::pow((x2 - x1 * x1), 2) + std::pow(1 - x1, 2);
  }
  void Gradient(const arma::vec& x, arma::vec& gr) override {
    gr = arma::zeros<arma::vec>(2);
    double x1 = x(0);
    double x2 = x(1);
    gr(0) = -4 * h * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
    gr(1) = 2 * h * (x2 - x1 * x1);
  }
  void Hessian(const arma::vec& x, arma::mat& he) override {
    he = arma::zeros<arma::mat>(2, 2);
    double x1 = x(0);
    double x2 = x(1);
    he(0, 0) = -4 * h * x2 + 1200 * x1 * x1 + 2;
    he(0, 1) = -4 * h * x1;
    he(1, 0) = he(0, 1);
    he(1, 1) = 2 * h;
  }
};

// [[Rcpp::export]]
void example1_rosen_bfgs() {
  Rosen rb;
  rb.h = 100;
  Roptim<Rosen> opt("BFGS");
  opt.control.trace = 1;
  opt.set_hessian(true);
  arma::vec x = {-1.2, 1};
  opt.minimize(rb, x);
  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();
}

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//
// [[Rcpp::export]]
arma::mat rcpparma_hello_world() {
  arma::mat m1 = arma::eye<arma::mat>(3, 3);
  arma::mat m2 = arma::eye<arma::mat>(3, 3);

  return m1 + 3 * (m1 + m2);
}

// another simple example: outer product of a vector,
// returning a matrix
//
// [[Rcpp::export]]
arma::mat rcpparma_outerproduct(const arma::colvec& x) {
  arma::mat m = x * x.t();
  return m;
}

// and the inner product returns a scalar
//
// [[Rcpp::export]]
double rcpparma_innerproduct(const arma::colvec& x) {
  double v = arma::as_scalar(x.t() * x);
  return v;
}

// and we can use Rcpp::List to return both at the same time
//
// [[Rcpp::export]]
Rcpp::List rcpparma_bothproducts(const arma::colvec& x) {
  arma::mat op = x * x.t();
  double ip = arma::as_scalar(x.t() * x);
  return Rcpp::List::create(Rcpp::Named("outer") = op,
                            Rcpp::Named("inner") = ip);
}
