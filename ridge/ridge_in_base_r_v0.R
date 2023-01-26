
setwd("~/PycharmProjects/ds-ml-algorithms")

set.seed(1409051)

n_lambdas <- 100

min_lambda <- 1
max_lambda <- 100

validation_split <- 0.8

# NOTE: Input data is already standardized.

df <- read.csv("./data/regression_df.csv")

val_index <- sample(1:nrow(df), round(validation_split * nrow(df)))

add_intercept <- function(X){
  return(as.matrix(cbind(rep(1, nrow(X)), X)))
}

ridge_equation <- function(XtX, Xty, lambda = 0){
  return(solve(XtX + lambda * diag(ncol(XtX))) %*% Xty)
}

get_rmse <- function(X, coefs, y){
  y_hat <- X %*% coefs
  return(sqrt((1 / nrow(X)) * sum((y_hat - y) ^ 2)))
}

get_lambdas <- function(){
  lambda_seq <- exp(seq(log(min_lambda), log(max_lambda), length.out = n_lambdas - 1))
  lambda_seq <- c(0, lambda_seq)
  return(lambda_seq)
}

run_ridge <- function(df, val_index){
  X_val <- add_intercept(df[val_index, -1])
  y_val <- as.matrix(df[val_index, 1])
  XtX_val <- t(X_val) %*% X_val
  Xty_val <- t(X_val) %*% y_val
  lambdas <- get_lambdas()
  ridge_coefs <- sapply(lambdas, function(l) ridge_equation(XtX_val, Xty_val, l))
  X_nonval <- add_intercept(df[-val_index, -1])
  y_nonval <- as.matrix(df[-val_index, 1])
  ridge_rmse <- sapply(1:ncol(ridge_coefs), function(j) get_rmse(X_nonval, ridge_coefs[,j], y_nonval))
  return(list(lambdas = lambdas, coefs = ridge_coefs, rmse = ridge_rmse))
}

ridge_result <- run_ridge(df, val_index)

lambda_index <- which.min(ridge_result$rmse)

lambda_value <- ridge_result$lambdas[lambda_index]

plot(x = ridge_result$lambdas, y = ridge_result$rmse, 
     xlab="Lambda", 
     ylab="RMSE",
     type="l")
abline(v=lambda_value, col = "red", lty = 2)

lambda_coefs <- ridge_result$coefs[,lambda_index]

matplot(x=ridge_result$lambda, y=t(ridge_result$coefs[-1,]), 
        type='l',
        xlab="Lambda",
        ylab="Coefficient value")
abline(v=lambda_value, col = "red", lty = 2)
