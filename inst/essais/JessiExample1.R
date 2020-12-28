library(gfilogisreg)

x <- rep(c(-2, -1, 0, 1, 2), times = 5L)
X <- model.matrix(~ x)
n <- length(x)
y <- c(matrix(c(
  1, 0, 0, 0, 0,
  1, 1, 1, 0, 0,
  1, 1, 0, 0, 0,
  1, 1, 1, 1, 0,
  1, 1, 1, 1, 1
), nrow = 5L, ncol = 5L, byrow = TRUE))

glm(y ~ x, family = binomial())

gf <- gfilogisreg(y ~ x, N = 10000,
                  ufactr = .Machine$double.eps^(-0.5),
                  vfactr = .Machine$double.eps^(-0.38))
gfiConfInt(~ -`(Intercept)`/x, gf)
