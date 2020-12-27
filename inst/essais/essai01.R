library(gfilogisreg)

ilogit <- function(a) exp(a) / (1+exp(a))

set.seed(666L)
n <- 50L
x <- seq(-3, 3, length.out = n)
beta <- c(0, 2)
probs <- ilogit(model.matrix(~x) %*% beta)
y <- rbinom(n, size = 1L, prob = probs)
gf <- gfilogisreg(y ~ x, N = 3000L)
gfiSummary(gf, conf = 0.9)

glm(y~x, family = binomial())

# stan ####
library(rstanarm)
options(mc.cores = parallel::detectCores())

bglm <- stan_glm(y ~ x, family = binomial(), data = data.frame(y=y, x=x),
                 iter = 10000)

posterior_interval(bglm)
