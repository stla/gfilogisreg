library(gfilogisreg)

ilogit <- function(a) exp(a) / (1+exp(a))

set.seed(666L)
n <- 51L
group <- gl(3L, 17L)
beta <- c(1, 2, 3)
probs <- ilogit(model.matrix(~ 0+group) %*% beta)
dat <- data.frame(
  group = group,
  y = rbinom(n, size = 1L, prob = probs)
)
gf <- gfilogisreg(y ~ 0 + group, data = dat, N = 3000L)
gfiSummary(gf, conf = 0.9)

glm(y ~ 0+group, data = dat, family = binomial())

# stan ####
library(rstanarm)
options(mc.cores = parallel::detectCores())

bglm <- stan_glm(y ~ 0+group, family = binomial(), data = dat,
                 iter = 10000)

posterior_interval(bglm)
