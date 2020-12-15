isone <- function(x){
  abs(x-1) < 0.1
}

#' @importFrom stats runif qlogis plogis
#' @noRd
rtlogis1 <- function(b){
  out <- qlogis(runif(1L, min = 0, max = plogis(b)))
  if(out == -Inf){
    if(b > qlogis(1e-16))
      out <- runif(1L, qlogis(1e-16), b)
    else
      out <- b
  }
  out
}
rtlogis2 <- function(b){
  out <- qlogis(runif(1L, min = plogis(b), max = 1))
  if(out == Inf){
    if(b < qlogis(1e-16, lower.tail = FALSE))
      out <- runif(1L, b, qlogis(1e-16, lower.tail = FALSE))
    else
      out <- b
  }
  out
}

#' @importFrom Runuran ur vnrou.new
#' @importFrom stats dlogis
#' @noRd
rcd <- function(n, P, b, B){
  ur(
    unr = vnrou.new(
      dim = length(B),
      pdf = function(x) prod(dlogis(c(P %*% x + b))),
      center = c(B)
    ),
    n = n
  )
}
