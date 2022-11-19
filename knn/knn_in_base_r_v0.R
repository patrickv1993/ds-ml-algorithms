
set.seed(1409051)

test_split <- 0.2

n_neighbors <- 5

# 2. We're going to use iris as the generic "df" for this,

df <- iris

# 2.1 Also store the row count/X, y columns for later
row_count <- nrow(df)
y_column <- "Species"
class_column <- "class"
neighbor_class_column <- "neighbor_class"
predicted_class_column <- "predicted_class"
X_columns <- names(iris)[names(iris) != y_column]

# 3. How to get the "nearest" neighbors?
# The common approach to KNN is to use euclidean distance as the metric of similarity.
# The "k" nearest neighbors in terms of this metric are what's used for classification.

# 3.1 Before we can get reasonable distance measurements, we need to standardize our data.

# Start by selecting columns, converting to matrix.
X <- as.matrix(df[X_columns])

# Next we calculate the column means and standard deviations.
column_means <- apply(X, 2, mean)
column_sds <- apply(X, 2, sd)

# Lastly we use the sweep() function to complete the standardization.
Z <- X
Z <- sweep(Z, 2, column_means, FUN = "-")
Z <- sweep(Z, 2, column_sds, FUN = "/")
rm(X) # Note, this is done to free up memory + make sure we don't overextend the purpose of X.

# 3.2 Calculating the distance matrix
# We can get the distance matrix with a simple call of the dist() function.
# Note two arguments I passed here to save memory / skip redundant calculations.

D <- dist(Z, method="euclidean", diag=FALSE, upper=FALSE)

# We can make it easier (but less efficient) if we store this as a matrix.
# But for this demo I am going to show how we can efficiently access data from the reduced form.

# The math behind it:
# A full distance matrix D has N * N elements.
# When we remove the upper triangle of a matrix, we drop N*(N - 1) / 2 elements.
# This leaves us with the classic "sum of integers" N*(N+1) / 2.
# When we subtract the diagonal, we are now left with N*(N+1) / 2 - N
# This calculation for our case yields 150 * 151 * (1 / 2) - 150 = 11175
# This should be the length of your data for the variable D.

# Now let's think about the algorithm part:
# If we look at the elements, we can see a pattern:
# 1. D[1] = (i=2, j=1)->1
# 2. D[N - 1] = (i=150, j=1)->149
# 3. D[(N - 1) + 1] = (i=3, j=2)->150
# 4. D[(N - 1) + (N - 2)] = (i=150, j=2)->297
# 5. D[(N - 1) + (N - 2) + 1] = (i=4, j=3)->298
# 6. D[(N - 1) + (N - 2) + (N - 3)] = (i=150, j=3)->444
# Now here's a general formula
# D[S_{k < j}(N - k) + i - j]
# Plugging this back in #1, we get (i=2, j=1) = 0 + 2 - 1 = 1
# For #2: (i=150, j=1) = 0 + 150 - 1 = 150 - 1
# For #3: (i=3, j=2) = (150 - 1) + 3 - 2 = 150
# For #4: (i=150, j=2) = (150 - 1) + (150 - 2) = 297
# For #5: (i=4, j=3) = (150 - 2) + (150 - 1) + 4 - 3 = 298
# For #6: (i=150, j=3)=(150 - 3) + (150 - 2) + (150 - 1) = 444

m_to_d_index <- function(i, j, N=row_count){
  return((j - 1) * N - sum(0:(j - 1)) + (i - j))
}

d_to_m_index <- function(k, N=row_count){
  i <- 1
  j <- 1
  for(l in 1:k){
    i <- i + 1
    if(i > N){
      j <- j + 1
      i <- j + 1
    }
  }
  return(c(i, j))
}

fetch_distance <- function(D, i, j, N=row_count){
  if(i == j){
    return(0)
  }
  if(i < j){
    # Note: R does not allow tuple assignment hence this is ugly.
    new_ij <- list(i = j, j = i)
    i = new_ij[["i"]]
    j = new_ij[["j"]]
    rm(new_ij)
  }
  k <- m_to_d_index(i, j, N)
  d <- D[k]
  return(d)
}

# 4. How to split the data?

# 4.1. Set an index variable that ranges 1:N, or pass in an index set if the data has one.
df_index <- 1:row_count

# 4.2. Sample 20% of the df_index values without replacement.
# That's your test df index. Also store the number of data points.

test_row_count <- round(row_count * test_split)
test_index <- sample(df_index, test_row_count, replace=FALSE)

# 4.3. Do the same thing for the training set
train_row_count <- row_count - test_row_count
train_index <- setdiff(df_index, test_index)

# iv. Last but not least, build an extra dataframe as a "tracking table"
# This will save us any headaches with indexing issues once we split the data.

index_map_df <- data.frame(
  original_index=c(test_index, train_index),
  new_index=c(1:test_row_count, 1:train_row_count),
  data_group=rep(c("test", "train"), times=c(test_row_count, train_row_count))
)

y_train <- df[train_index, y_column]
y_test <- df[test_index, y_column]
y <- c(y_test, y_train)

index_map_df[[y_column]] <- y

# Also, order the dataframe and remove the old rownames
index_map_df <- index_map_df[order(index_map_df$original_index),]
rownames(index_map_df) <- NULL

# One more note on this!! For larger df's, it might make sense to just merge
# this table back onto the original one. With the Iris dataset being 150 rows it's no big deal.

# 5. Second to last, we want to get the nearest neighbor data!

fetch_neighbor_df <- function(
    index_map_df,
    D,
    j,
    k=n_neighbors,
    data_group="train",
    filter_k=TRUE){
  index_filter <- index_map_df["original_index"] == j
  data_group_filter <- index_map_df["data_group"] == data_group
  df_slice <- index_map_df[index_filter,]
  new_index <- df_slice[["new_index"]][1]
  data_group_slice <- df_slice[["data_group"]][1]
  y_slice <- df_slice[[y_column]][1]

  df_other_slice <- index_map_df[!index_filter & data_group_filter,]
  df_other_slice$distance <- sapply(df_other_slice$original_index, function(i)fetch_distance(D, i, j))
  df_other_slice <- df_other_slice[order(df_other_slice$distance, decreasing=FALSE),]
  df_other_slice$order <- 1:nrow(df_other_slice)

  if(filter_k){
    df_other_slice <- df_other_slice[df_other_slice["order"] <= k,]
  }

  names(df_other_slice) <- gsub(y_column, class_column, names(df_other_slice))
  names(df_other_slice) <- paste("neighbor", names(df_other_slice), sep="_")

  neighbor_df <- data.frame(
    original_index=j,
    new_index=new_index,
    data_group=data_group_slice
  )
  neighbor_df[[class_column]] <- y_slice

  neighbor_df <- cbind(neighbor_df, df_other_slice)

  return(neighbor_df)
}

neighbor_df_tall <- do.call(rbind, lapply(index_map_df$original_index,function(j)fetch_neighbor_df(index_map_df, D, j)))

# 6. Last part of the algorithm is predicting the class of a point given the classes of nearest neighbors.

decision_function <- function(vector_of_classes){
  class_counts <- table(vector_of_classes)
  chosen_classes <- names(class_counts)[class_counts == max(class_counts)]
  chosen_class <- sample(chosen_classes, 1, replace=FALSE)
  return(chosen_class)
}

get_prediction_df_slice <- function(neighbor_df_tall, index){
  neighbor_df_slice <- neighbor_df_tall[neighbor_df_tall["original_index"] == index,]
  predicted_class <- decision_function(neighbor_df_slice[neighbor_class_column])
  prediction_df_slice <- data.frame(
    original_index=index,
    new_index=neighbor_df_slice[["new_index"]][1],
    data_group=neighbor_df_slice[["data_group"]][1],
    class_column=neighbor_df_slice[[class_column]][1],
    predicted_class_column=predicted_class
  )
  return(prediction_df_slice)
}

prediction_df <- do.call(rbind, lapply(index_map_df$original_index,
                                       function(index) get_prediction_df_slice(neighbor_df_tall, index)))

# 7. Lastly we can get summary stats based on train/test splits.

table(prediction_df$class_column,
      prediction_df$predicted_class_column,
      prediction_df$data_group)