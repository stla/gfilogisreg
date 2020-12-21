library(numDeriv)


P <- structure(c(-0.816496580927726, -0.408248290463863, 0, 0.408248290463863,
                 -0.182574185835055, -0.365148371670111, -0.547722557505166, -0.730296743340222
), .Dim = c(4L, 2L))
b <- c(-0.497055737731043, 1.5763637399835, -1.66156026677388,
       0.582252264521417)
f <- function(uv){
  vecx <- P %*% logit(uv) + b
  prod(dlogis(vecx)) * prod(dlogit(uv))
}
minus_logf <- function(uv){
  vecx <- P %*% logit(uv) + b
  -sum(ldlogis(vecx)) - sum(ldlogit(uv))
}
gr_i <- function(uv, i){
  vecx <- P %*% logit(uv) + b
  -dlogit(uv[i]) * sum(P[, i] * dldlogis(vecx)) + (1-2*uv[i])/(uv[i]*(1-uv[i]))
}
gr <- function(uv){
  c(gr_i(uv, 1L), gr_i(uv, 2L))
}

# umax ####
B <- c(0.456112464221194, 0.206538736468621)
opt <- optim(
  par = expit(B), fn = minus_logf,  gr = gr,
  method = "L-BFGS-B", lower = c(1e-16, 1e-16), upper = c(1-1e-16, 1-1e-16)
)
mu <- opt[["par"]]
umax <- sqrt(exp(-minus_logf(mu)))

# vmin ####
vmin <- c(NA_real_, NA_real_)
#
init <- expit(B)
init[1] <- mu[1]/2
opt <- optim(
  par = init,
  fn = function(uv) (exp(-minus_logf(uv)))^0.25 * (uv[1] - mu[1]),
  #gr = function(uv) -gr(uv) + c(1/uv[1], 0),
  method = "L-BFGS-B",
  lower = c(1e-16, 1e-16),
  upper = c(mu[1], 1-1e-16)
)
vmin[1] <- opt[["value"]]
#
init <- expit(B)
init[2] <- mu[2]/2
opt <- optim(
  par = init,
  fn = function(uv) (exp(-minus_logf(uv)))^0.25 * (uv[2] - mu[2]),
  #gr = function(uv) -gr(uv) + c(1/uv[1], 0),
  method = "L-BFGS-B",
  lower = c(1e-16, 1e-16),
  upper = c(1-1e-16, mu[2])
)
vmin[2] <- opt[["value"]]

# vmax ####
vmax <- c(NA_real_, NA_real_)
#
init <- expit(B)
init[1] <- (mu[1]+1)/2
opt <- optim(
  par = init,
  fn = function(uv) -f(uv) * (uv[1] - mu[1])^4,
  #gr = function(uv) -gr(uv) + c(1/uv[1], 0),
  method = "L-BFGS-B",
  lower = c(mu[1], 1e-16),
  upper = c(1-1e-16, 1-1e-16)
)
vmax[1] <- (-opt[["value"]])^0.25
#
init <- expit(B)
init[2] <- (mu[2]+1)/2
opt <- optim(
  par = init,
  fn = function(uv) -exp(-minus_logf(uv)) * (uv[2] - mu[2])^4,
  #gr = function(uv) -gr(uv) + c(1/uv[1], 0),
  method = "L-BFGS-B",
  lower = c(1e-16, mu[2]),
  upper = c(1-1e-16, 1-1e-16)
)
vmax[2] <- (-opt[["value"]])^0.25
