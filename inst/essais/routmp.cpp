#include "RcppArmadillo.h"
#include "roptim.h"
using namespace roptim;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]

std::vector<size_t> CantorExpansion(size_t n, std::vector<size_t> s){
  std::vector<size_t> out(s.size());
  std::vector<size_t>::iterator it;
  it = s.begin();
  it = s.insert ( it , 1 );
  size_t G[s.size()];
  std::partial_sum (s.begin(), s.end(), G, std::multiplies<size_t>());
  size_t k;
  while(n>0){
    k=1;
    while(G[k]<=n){
      k++;
    }
    out[k-1] = n / G[k-1];
    n = n % G[k-1];
  }
  return out;
}

// [[Rcpp::export]]
arma::mat grid(const size_t d){
  std::array<double, 3> x = {0.01, 0.5, 0.99};
  size_t p = pow((size_t)3, d);
  arma::mat out(d, p);
  std::vector<size_t> threes(d, 3);
  for(size_t n = 0; n < p; n++){
    std::vector<size_t> indices = CantorExpansion(n, threes);
    for(size_t i = 0; i < d; i++){
      out(i, n) = x[indices[i]];
    }
  }
  return out;
}

arma::vec logit(const arma::vec& u) {
  return arma::log(u) - arma::log(1.0 - u);
}

double dlogit(double u){
  return 1.0 / (u*(1.0-u));
}

arma::vec ldlogit(const arma::vec& u) {
  return -arma::log(u) - arma::log1p(-u);
}

arma::vec ldlogis(const arma::vec& x) {
  return x - 2.0 * arma::log1p(arma::exp(x));
}

arma::vec dldlogis(const arma::vec& x) {
  return 1.0 - 2.0 / (1.0 + arma::exp(-x));
}

double log_f(const arma::vec &u, const arma::mat& P, const arma::vec& b){
  const arma::vec x = P * logit(u) + b;
  return arma::sum(ldlogis(x)) + arma::sum(ldlogit(u));
}

double dlog_f(const double ui, const arma::vec& Pi, const arma::vec& y){
  return dlogit(ui) * arma::sum(Pi % y) + (2.0*ui-1.0) / (ui * (1-ui));
}

class Logf : public Functor {
public:
  arma::mat P;
  arma::vec b;
  size_t d;
  double operator()(const arma::vec &u) override {
    return log_f(u, P, b);
  }
  void Gradient(const arma::vec &u, arma::vec &gr) override {
    gr = arma::zeros<arma::vec>(d);
    const arma::vec y = dldlogis(P * logit(u) + b);
    for(size_t i = 0; i < d; i++){
      gr(i) = dlog_f(u[i], P.col(i), y);
    }
  }
};

Rcpp::List get_umax0(const arma::mat& P, const arma::vec& b, arma::vec B) {
  double eps = std::numeric_limits<double>::epsilon();
  Logf logf;
  logf.P = P; logf.b = b; logf.d = B.size();
  Roptim<Logf> opt("L-BFGS-B");
  opt.control.trace = 1;
  opt.control.maxit = 1000;
  opt.control.fnscale = -1.0; // maximize
  opt.control.factr = 1.0;
  opt.set_hessian(false);
  arma::vec lwr = arma::zeros(B.size()) + eps;
  arma::vec upr = arma::ones(B.size()) - eps;
  opt.set_lower(lwr); opt.set_upper(upr);
  opt.minimize(logf, B);
  Rcpp::Rcout << "-------------------------" << std::endl;
  //  opt.print();
  return Rcpp::List::create(
    Rcpp::Named("par") = opt.par(),
    Rcpp::Named("value") = opt.value()
  );
}

// [[Rcpp::export]]
Rcpp::List get_umax(const arma::mat& P, const arma::vec& b, arma::mat& Bs){
  const size_t n = Bs.n_cols;
  const size_t d = Bs.n_rows;
  std::vector<arma::vec> pars(n);
  arma::vec values(n);
  for(size_t i = 0; i < n; i++){
    const Rcpp::List L = get_umax0(P, b, Bs.col(i));
    const arma::vec par = L["par"];
    pars[i] = par;
    //double value = L["value"];
    values(i) = L["value"];
  }
  const size_t imax = values.index_max();
  return Rcpp::List::create(
    Rcpp::Named("mu") = pars[imax],
                            Rcpp::Named("umax") = pow(exp(values(imax)), 2.0/(2.0 + d))
  );
}

class Rosen : public Functor {
public:
  double h;
  double operator()(const arma::vec &x) override {
    double x1 = x(0);
    double x2 = x(1);
    return h * std::pow((x2 - x1 * x1), 2) + std::pow(1 - x1, 2);
  }
  void Gradient(const arma::vec &x, arma::vec &gr) override {
    gr = arma::zeros<arma::vec>(2);
    double x1 = x(0);
    double x2 = x(1);
    gr(0) = -4*h * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
    gr(1) = 2*h * (x2 - x1 * x1);
  }
  void Hessian(const arma::vec &x, arma::mat &he) override {
    he = arma::zeros<arma::mat>(2, 2);
    double x1 = x(0);
    double x2 = x(1);
    he(0, 0) = -4*h * x2 + 1200 * x1 * x1 + 2;
    he(0, 1) = -4*h * x1;
    he(1, 0) = he(0, 1);
    he(1, 1) = 2*h;
  }
};

// [[Rcpp::export]]
void example1_rosen_bfgs()
{
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
