// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <boost/multiprecision/gmp.hpp>

namespace mp = boost::multiprecision;

// [[Rcpp::export]]
double testgmp(){
  mp::mpq_rational q("3/4");
  double x = q.convert_to<double>();
  return x;
}

// [[Rcpp::export]]
double testgmp2(std::string s){
  mp::mpq_rational q(s);
  double x = q.convert_to<double>();
  return x;
}

// [[Rcpp::export]]
Rcpp::CharacterVector testgmp3(Rcpp::NumericVector x){
  Rcpp::CharacterVector out(x.size());
  for(auto i = 0; i < x.size(); i++){
    mp::mpq_rational q(x(i));
    out(i) = q.convert_to<std::string>();
  }
  return out;
}
