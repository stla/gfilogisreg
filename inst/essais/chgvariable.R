P <- structure(c(-0.816496580927726, -0.408248290463863, 0, 0.408248290463863,
                 -0.182574185835055, -0.365148371670111, -0.547722557505166, -0.730296743340222
), .Dim = c(4L, 2L))
b <- c(-0.497055737731043, 1.5763637399835, -1.66156026677388,
       0.582252264521417)
b <- c(-0.549834236699787, 0.408894029413032, 0.831714651273296,
       -0.690774443986541)

g <- function(x) atan(x)/pi + 0.5#x/(1+x)
h <- function(u) tan(pi*(u-0.5))#   u/(1-u)
g(h(0.3))
dh <- function(u) pi/cos(pi*(u-0.5))^2
h <- function(u) log(u/(1-u))
dh <- function(u) 1/(u*(1-u))

f <- function(uv){
  vecx <- P %*% h(uv) + b
  prod(dlogis(vecx)) #* prod(dh(uv))
}
logf <- function(uv){
  log(f(uv))
}

library(graph3d)
dat <- expand.grid(
  x = seq(0.01,0.99,length.out=50),
  y = seq(0.01,0.99,length.out=50)
)
dat$z <- apply(dat, 1, f)
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
