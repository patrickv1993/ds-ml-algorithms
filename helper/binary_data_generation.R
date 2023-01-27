
setwd("~/PycharmProjects/ds-ml-algorithms")

set.seed(1409051)

N <- 10000

U1 <- runif(N, -1, 1)
U2 <- runif(N, -1, 1)
t <- sort(rbeta(2, 2, 2))
y <- (t[1] <= U1^2 + U2^2) & (U1 ^2 + U2 ^2 <= t[2])

df <- data.frame(U1, U2, y)

write.csv(df, "./data/binary_df.csv", row.names = FALSE)