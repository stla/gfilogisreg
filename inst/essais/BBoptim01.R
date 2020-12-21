library(BB)

expit <- function(x) 1 / (1+exp(-x))
logit <- function(u) log(u/(1-u))
dlogit <- function(u) 1/(u*(1-u))
ldlogit <- function(u) -log(u) - log1p(-u)
ldlogis <- function(x) x - 2*log1p(exp(x))
dldlogis <- function(x) 1 - 2*expit(x)

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
grl_i <- function(uv, i){
  vecx <- P %*% logit(uv) + b
  dlogit(uv[i]) * sum(P[, i] * dldlogis(vecx)) + (2*uv[i]-1)/(uv[i]*(1-uv[i]))
}
grl <- function(uv){
  c(grl_i(uv, 1L), grl_i(uv, 2L))
}
gr <- function(uv){
  f(uv) * grl(uv)
}

# for the initial values
B <- c(0.456112464221194, 0.206538736468621)


# umax ####
opt <- BBoptim(
  par = expit(B),
  fn = f,
  gr = gr,
  lower = c(1e-16, 1e-16),
  upper = c(1-1e-16, 1-1e-16),
  control = list(maximize = TRUE, trace = FALSE)
)
opt$convergence
mu <- opt[["par"]]
umax <- sqrt(opt[["value"]])

# vmin ####
vmin <- c(NA_real_, NA_real_)
#
init <- expit(B)
init[1] <- mu[1]/2
opt <- BBoptim(
  par = init,
  fn = function(uv) -(-minus_logf(uv) + 4*log(mu[1] - uv[1])),
  gr = function(uv) -grl(uv) + 4 * c(1/(mu[1] - uv[1]), 0),
  lower = c(1e-16, 1e-16),
  upper = c(mu[1]-1e-16, 1-1e-16),
  control = list(maximize = FALSE, trace = FALSE)
)
opt$convergence
vmin[1] <- -exp(-opt[["value"]]/4) # f(opt[["par"]])^0.25 * (opt[["par"]][1] - mu[1])
#
init <- expit(B)
init[2] <- mu[2]/2
opt <- BBoptim(
  par = init,
  fn = function(uv) -(-minus_logf(uv) + 4*log(mu[2] - uv[2])),
  gr = function(uv) -grl(uv) + 4 * c(0, 1/(mu[2] - uv[2])),
  lower = c(1e-16, 1e-16),
  upper = c(1-1e-16, mu[2]-1e-16),
  control = list(maximize = FALSE, trace = FALSE)
)
opt$convergence
vmin[2] <- -exp(-opt[["value"]]/4) # f(opt[["par"]])^0.25 * (opt[["par"]][2] - mu[2])

# vmax ####
vmax <- c(NA_real_, NA_real_)
#
init <- expit(B)
init[1] <- (mu[1]+1)/2
opt <- BBoptim(
  par = init,
  fn = function(uv) -minus_logf(uv) + 4*log(uv[1] - mu[1]),
  gr = function(uv) grl(uv) - 4 * c(1/(mu[1] - uv[1]), 0),
  lower = c(mu[1]+1e-16, 1e-16),
  upper = c(1-1e-16, 1-1e-16),
  control = list(maximize = TRUE, trace = FALSE)
)
opt$par
vmax[1] <- exp(opt[["value"]]/4) # f(opt[["par"]])^0.25 * (opt[["par"]][1] - mu[1])
#
init <- expit(B)
init[2] <- (mu[2]+1)/2
opt <- BBoptim(
  par = init,
  fn = function(uv) -minus_logf(uv) + 4*log(uv[2] - mu[2]),
  gr = function(uv) grl(uv) - 4 * c(0, 1/(mu[2] - uv[2])),
  lower = c(1e-16, mu[2]+1e-16),
  upper = c(1-1e-16, 1-1e-16),
  control = list(maximize = TRUE, trace = FALSE)
)
vmax[2] <- exp(opt[["value"]]/4)


# simulations ####
nsims <- 3000L

sims <- matrix(NA_real_, nrow = nsims, ncol = 2)
k <- 0L
while(k < nsims){
  u <- runif(1L, 0, umax)
  v <- runif(2L, vmin, vmax)
  x <- v/sqrt(u) + mu
  test <- all(x > 0) && all(x < 1) && u < sqrt(f(x))
  if(test){
    k <- k + 1L
    sims[k, ] <- x
  }
}

sims2 <- gfilogisreg:::rcd(nsims, P, b, B)

plot(logit(sims[,1]), logit(sims[,2]))
points(sims2[,1], sims2[,2], col = "red")
