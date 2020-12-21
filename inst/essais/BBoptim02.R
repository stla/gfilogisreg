library(BB)

expit <- function(x) 1 / (1+exp(-x))
logit <- function(u) log(u/(1-u))
dlogit <- function(u) 1/(u*(1-u))
ldlogit <- function(u) -log(u) - log1p(-u)
ldlogis <- function(x) x - 2*log1p(exp(x))
dldlogis <- function(x) 1 - 2*expit(x)

rcd <- function(n, P, b, B){
  d <- length(B)
  f <- function(uv){
    vecx <- P %*% logit(uv) + b
    prod(dlogis(vecx)) * prod(dlogit(uv))
  }
  logf <- function(uv){
    vecx <- P %*% logit(uv) + b
    sum(ldlogis(vecx)) + sum(ldlogit(uv))
  }
  grl_i <- function(uv, i){
    vecx <- P %*% logit(uv) + b
    dlogit(uv[i]) * sum(P[, i] * dldlogis(vecx)) + (2*uv[i]-1)/(uv[i]*(1-uv[i]))
  }
  grl <- function(uv){
    vapply(1L:d, function(i) grl_i(uv, i), numeric(1L))
  }
  gr <- function(uv){
    f(uv) * grl(uv)
  }
  # umax ####
  opt <- BBoptim(
    par = expit(B),
    fn = f,
    gr = gr,
    lower = rep(1e-16, d),
    upper = rep(1-1e-16, d),
    control = list(maximize = TRUE, trace = FALSE)
  )
  mu <- opt[["par"]]
  umax <- sqrt(opt[["value"]])
  # vmin ####
  vmin <- numeric(d)
  for(i in 1L:d){ # !!!! rd !!!!! ici d=2 uniquement !!!
    opt <- BBoptim(
      par = `[<-`(rep(0.5, d), i, mu[i]/2),
      fn = function(uv) -(logf(uv) + 4*log(mu[i] - uv[i])),
      gr = function(uv) -grl(uv) + 4 * `[<-`(numeric(d), i, 1/(mu[i] - uv[i])),
      lower = rep(1e-16, d),
      upper = `[<-`(rep(1, d), i, mu[i]) - 1e-16,
      control = list(maximize = FALSE, trace = FALSE)
    )
    vmin[i] <- -exp(-opt[["value"]]/4)
  }
  # vmax ####
  vmax <- numeric(d)
  for(i in 1L:d){ # !!!! rd !!!!! ici d=2 uniquement !!!
    opt <- BBoptim(
      par = `[<-`(rep(0.5, d), i, (mu[i]+1)/2),
      fn = function(uv) logf(uv) + 4*log(uv[i] - mu[i]),
      gr = function(uv) grl(uv) - 4 * `[<-`(numeric(d), i, 1/(mu[i] - uv[i])),
      lower = `[<-`(numeric(d), i, mu[i]) + 1e-16,
      upper = rep(1-1e-16, d),
      control = list(maximize = TRUE, trace = FALSE)
    )
    vmax[i] <- exp(opt[["value"]]/4)
  }
  # simulations
  sims <- matrix(NA_real_, nrow = n, ncol = d)
  k <- 0L
  while(k < n){
    u <- runif(1L, 0, umax)
    v <- runif(d, vmin, vmax)
    x <- v/sqrt(u) + mu
    if(all(x > 0) && all(x < 1) && u < sqrt(f(x))){
      k <- k + 1L
      sims[k, ] <- x
    }
  }
  logit(sims)
}


P <- structure(c(-0.816496580927726, -0.408248290463863, 0, 0.408248290463863,
                 -0.182574185835055, -0.365148371670111, -0.547722557505166, -0.730296743340222
), .Dim = c(4L, 2L))
b <- c(-0.497055737731043, 1.5763637399835, -1.66156026677388,
       0.582252264521417)
# for the initial value
B <- c(0.456112464221194, 0.206538736468621)

# simulations ####
nsims <- 30L
sims <- rcd(nsims, P, b, B)
sims2 <- gfilogisreg:::rcd(nsims, P, b, B)

plot(sims[,1], sims[,2])
points(sims2[,1], sims2[,2], col = "red")
