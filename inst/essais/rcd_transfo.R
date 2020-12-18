library(Runuran)

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

logit <- function(u) log(u/(1-u))
dlogit <- function(u) 1/(u*(1-u))
expit <- function(x) exp(x) / (1+exp(x))

rcd2 <- function(n, P, b, expitB){
  d <- length(expitB)
  ur(
    unr = vnrou.new(
      dim = d,
      pdf = function(u) prod(dlogis(c(P %*% logit(u) + b))) * prod(dlogit(u)),
      center = c(expitB),
      ll = rep(0, d), ur = rep(1,d)
    ),
    n = n
  )
}

P <- cbind(c(2,1,2), c(-2,2,1))/3
b <- c(1,2,-2)/3

B <- c(0,0)
sims1 <- rcd(150000, P, b, B)
sims2 <- logit(rcd2(150000, P, b, expit(B)))

colMeans(sims1)
colMeans(sims2)
apply(sims1, 2, median)
apply(sims2, 2, median)

plot(sims1[,1], sims1[,2], pch=19)
points(sims2[,1], sims2[,2], col = "red")


