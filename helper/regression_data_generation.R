
# Use this script to generate data for regression problems

setwd("~/PycharmProjects/ds-ml-algorithms")

library(MASS)

set.seed(1409051)

N <- 10000

intercept <- 500

k1 <- 10

k2 <- 180

k3 <- 10

# 20 "weak signal" variables

X_weak <- matrix(runif(N * k1, -sqrt(12) / 2, sqrt(12) / 2), nrow = N)

# 20 "highly correlated signal" variables

correlations <- rep(c(0.999, -0.999), each = k2 / 2)
Sigma <- correlations %*% t(correlations)
diag(Sigma) <- 1

X_strong_corr <- mvrnorm(N, rep(0, k2), Sigma)

# 10 "uncorrelated signal" variables

X_strong_no_corr <- matrix(runif(N * k3, -sqrt(12) / 2, sqrt(12) / 2), nrow = N)

# 1 error term

coefs <- c(rep(c(0.1, -0.1), each = k1 / 2), 
           rep(c(1, -1), each = k2 / 2), 
           rep(c(5, -5), each = k3 / 2))

error_term <- rnorm(N, sd = k1 + k2 + k3)

X <- cbind(X_weak, X_strong_corr, X_strong_no_corr)

y <- intercept + X %*% coefs + error_term

X_index <- sample(1:ncol(Z))

column_df <- data.frame(observed_index=paste0("X", X_index), 
                        true_index=paste0("X", 1:ncol(X)), 
                        var_type=rep(c("weak", "strong_correlated", "strong_uncorrelated"), 
                                     times=c(k1, k2, k3)))
column_df <- column_df[order(column_df$observed_index),]

df <- data.frame(y = y, X = X[, X_index])

names(df)[-1] <- gsub("\\.", "", names(df)[-1])

write.csv(column_df, "./data/regression_df_columns.csv", row.names = FALSE)

write.csv(df, "./data/regression_df.csv", row.names = FALSE)