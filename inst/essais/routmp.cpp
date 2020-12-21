#include "RcppArmadillo.h"
#include "roptim.h"
using namespace roptim;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(roptim)]]

arma::vec logit(arma::vec& u) {
  return arma::log(u) - arma::log(1 - u);
}

arma::vec ldlogit(arma::vec& u) {
  return -arma::log(u) - arma::log1p(1 - u);
}

arma::vec ldlogis(arma::vec& x) {
  return x - 2.0 * arma::log1p(arma::exp(x));
}

arma::vec dldlogis(arma::vec& x) {
  return 1.0 - 2.0 / (1.0 + arma::exp(-x));
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
