P <- structure(c(-0.816496580927726, -0.408248290463863, 0, 0.408248290463863,
                 -0.182574185835055, -0.365148371670111, -0.547722557505166, -0.730296743340222
), .Dim = c(4L, 2L))
b <- c(-0.497055737731043, 1.5763637399835, -1.66156026677388,
       0.582252264521417)
b <- c(-0.549834236699787, 0.408894029413032, 0.831714651273296,
       -0.690774443986541)
d <- ncol(P)

g <- function(x) atan(x)/pi + 0.5
h <- function(u) tan(pi*(u-0.5))
g(h(0.3))
dh <- function(u) pi/cos(pi*(u-0.5))^2

ldlogis <- function(x) x - 2*log1p(exp(x))
expit <- function(x) 1 / (1+exp(-x))
dldlogis <- function(x) 1 - 2*expit(x)

f <- function(uv){
  vecx <- P %*% h(uv) + b
  prod(dlogis(vecx))
}
# logf <- function(uv){
#   vecx <- P %*% h(uv) + b
#   sum(ldlogis(vecx))
# }
# grl_i <- function(uv, i){
#   vecx <- P %*% h(uv) + b
#   dh(uv[i]) * sum(P[, i] * dldlogis(vecx))
# }
grf_i <- function(uv, i){
  vecx <- P %*% h(uv) + b
  prod(dlogis(vecx)) * dh(uv[i]) * sum(P[, i] * dldlogis(vecx))
}
grf <- function(uv){
  vapply(1L:d, function(i) grf_i(uv, i), numeric(1L))
}
grxf_i <- function(uv, i, j, mu){
  d <- length(mu)
  alpha <- 1 / (d + 2)
  vecx <- P %*% h(uv) + b
  dfalpha <- alpha * prod(dlogis(vecx))^alpha * dh(uv[i]) * sum(P[, i] * dldlogis(vecx))# alpha * grf_i(uv, i) * f(uv)^(alpha-1)
  # vecx <- P %*% h(uv) + b
  if(i == j){
    dfalpha  * (h(uv[i]) - mu[i]) + f(uv)^alpha * dh(uv[i])
    # prod(dlogis(vecx)) *
    #   (dh(uv[i]) * (sum(P[, i] * dldlogis(vecx)) * (h(uv[i]) - mu[i]) + 1))
  }else{
    dfalpha * (h(uv[j]) - mu[j])
  }
}
grxf <- function(uv, mu, j){
  vapply(1L:d, function(i) grxf_i(uv, i, j, mu), numeric(1L))
}

# umax ####
opt <- optim(
  par = rep(0.5, d),
  fn = f,
  gr = grf,
  lower = rep(0, d),
  upper = rep(1, d),
  control = list(fnscale = -1, factr = 1),
  method = "L-BFGS-B"
)
mu <- h(opt[["par"]])
umax <- opt[["value"]]^(2/(d+2))

# vmin i
i <- 1
BBoptim(
  par = `[<-`(rep(0.5, d), i, g(mu[i])/2),
  fn = function(uv) f(uv)^(1/(d+2)) * (h(uv[i]) - mu[i]),
  gr = function(uv) grxf(uv, mu, i),
  lower = rep(0, d),
  upper = `[<-`(rep(1, d), i, g(mu[i]))
  # control = list(factr = 1),
  # method = "L-BFGS-B"
)
optim(
  par = `[<-`(rep(0.5, d), i, g(mu[i])/2),
  fn = function(uv) f(uv)^(1/(d+2)) * (h(uv[i]) - mu[i]),
  gr = function(uv) grxf(uv, mu, i),
  lower = rep(0, d),
  upper = `[<-`(rep(1, d), i, g(mu[i])),
  control = list(factr = 1),
  method = "L-BFGS-B"
)

# vmax i
i <- 1
BBoptim(
  par = `[<-`(rep(0.5, d), i, (g(mu[i])+1)/2),
  fn = function(uv) f(uv)^(1/(d+2)) * (h(uv[i]) - mu[i]),
  gr = function(uv) grxf(uv, mu, i),
  lower = `[<-`(rep(0, d), i, g(mu[i])),
  upper = rep(1, d),
  control = list(maximize = TRUE),
  # method = "L-BFGS-B"
)
optim(
  par = `[<-`(rep(0.5, d), i, (g(mu[i])+1)/2),
  fn = function(uv) f(uv)^(1/(d+2)) * (h(uv[i]) - mu[i]),
  gr = function(uv) grxf(uv, mu, i),
  lower = `[<-`(rep(0, d), i, g(mu[i])),
  upper = rep(1, d),
  control = list(fnscale = -1, factr = 1),
  method = "L-BFGS-B"
)




library(graph3d)
dat <- expand.grid(
  x = seq(0.01,0.99,length.out=50),
  y = seq(0.01,0.99,length.out=50)
)
dat$z <- apply(dat, 1, logf)
graph3d(dat, z = ~z, keepAspectRatio = FALSE, verticalRatio = 1)
f(c(1-2e-16, 1-2e-16))

xf <- function(uv){
  (uv[1]-0.5)*f(uv)
}
dat <- expand.grid(
  x = seq(0.01,0.99,length.out=50),
  y = seq(0.01,0.99,length.out=50)
)
dat$z <- apply(dat, 1, xf)
graph3d(dat, z = ~z, keepAspectRatio = FALSE, verticalRatio = 1)
