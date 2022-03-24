# BTC1899 - Deep Learning in Health
# Tutorial Week 7
# Date: March 3, 2022
# By Nicholas Mitsakakis

# DLNN in Practice Part 2

# More examples of multi-layer NN

# Clear the environment - BE CAREFUL!
rm(list = ls())

# Load packages
library(keras)
library(ggplot2)

#############################################################
###### Case study 2: multiclass classification example ######

# loading the Rueters dataset

reuters <- dataset_reuters(num_words = 10000)

train_data <- reuters$train$x
train_labels <- reuters$train$y
test_data <- reuters$test$x
test_labels <- reuters$test$y

str(train_data)
length(train_data)
length(test_data)

# format of data
train_data[[1]]

# similar with IMBD data

# decoding back to the newswire
word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

# it seems (as it is mentoned in the Chollet and Allard book) that integers have also shifted by 3, in this data representation, to allow for 0 to be for padding, 1 for start of the sequence and 2 for unknown word

my.fun <- function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
}

# in this function, index needs to be subtracted by 3 to give the correct representation

decoded_newswire <- sapply(train_data[[1]], my.fun)
paste(decoded_newswire, collapse = " ")


###############
# now we are building the representation of each review as a binary vector of length 10000

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

str(x_train)
sum(x_train[1,])
sum(x_train[2,])
# sparse data

#####################################
# Processing the labels

range(train_labels)
# from 0 to 45, we need to add 1 for one-hot encoding

# It seems that the topics correspond to:
topics <- c('cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply', 'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas','cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin', 'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',  'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead')

# check the topic of the first entry
topics[train_labels[1]+1]

# Another example
decoded_newswire <- sapply(train_data[[10]], my.fun)
paste(decoded_newswire, collapse = " ")
topics[train_labels[10]+1]


# made function for one-hot encoding
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels))
    results[i, labels[[i]] + 1] <- 1 # we add 1 here
  results
}

one_hot_train_labels <- to_one_hot(train_labels)          
one_hot_test_labels <- to_one_hot(test_labels)            

# alternatively, using built in functions for keras
y_train <- to_categorical(train_labels)
y_test <- to_categorical(test_labels)

# confirming it worked
y_train[1,]
train_labels[1]



# Defining the network
# Since the output layer has 46 units, we want the previous layers to be large enough
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

# Compiling the network, using categorical_crossentropy
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Setting up a validation set
set.seed(123)
val_indices <- sample(nrow(x_train),1000)

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices,]
partial_y_train <- y_train[-val_indices,]

# Training and validating the model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)


# It seems that validation accuracy gets worse after 9 epochs
# So we retrain a model with 9 epochs
model2 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

# Compiling the network, using categorical_crossentropy
model2 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Training and validating the model
history <- model2 %>% fit(
  x_train,
  y_train,
  epochs = 9,
  batch_size = 512
)

results <- model2 %>% evaluate(x_test, y_test)

# looking at the predictions, and comparing with the classes
preds <- predict(model2, x_test)
preds.cl <- max.col(preds)
table(max.col(y_test),preds.cl)


# How good are these accuracy results?
# Empirical distribution of the accuracy of a "random" classifier on these data that preserves the class proportions 

N <- 1000
random.acc <- c()
for (i in 1:N){
  tmp.perm <- sample(test_labels)
  random.acc[i] <- mean(tmp.perm == test_labels)
}
hist(random.acc)

# it is around 0.19, so 0.79 is a good accuracy in this dataset

# if we ignore the expected distribution, and assume uniform, 
# we get much worse results
N <- 100
random.acc <- c()
for (i in 1:N){
#  tmp.perm <- sample(test_labels)
  tmp.perm <- sample.int(46,length(test_labels),replace = T)-1
  random.acc[i] <- mean(tmp.perm == test_labels)
}
hist(random.acc)


##############################
## CASE STUDY 3: REGRESSION ##

# Boston housing dataset

boston.data <- dataset_boston_housing()

train_data <- boston.data$train$x
train_targets <- boston.data$train$y
test_data <- boston.data$test$x
test_targets <- boston.data$test$y


# The targets are the median values of owner-occupied homes, in thousands of dollars.

# Features: per capita crime rate, average number of rooms per dwelling, accessibility to highways, etc.

str(train_data)
str(test_data)

str(train_targets)
str(test_targets)

boxplot(train_data)
# big discrepancies in the scales, scaling is recommended

# Scaling the data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)

# scaling the test data using mean and std values from the training data
test_data <- scale(test_data, center = mean, scale = std)

# Writing a function to perform the first steps of model building
# This function can potentially use arguments to determing the tuning parameters of the network, e.g. applying regularization yes/no, etc

# Here we do not use tuning, so we have no arguments
build_model <- function() {                                1
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

# Using 5-fold cross validation 
k <- 5
indices <- sample(1:nrow(train_data))
# using the cut function to create the folds
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 100
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 16, verbose = 0)
  
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results['mae'])
}

all_scores
mean(all_scores)
# Mean Absolute Deviation (MAD) from the data:
mean(abs(train_targets-mean(train_targets)))
# so the performance is about 37% of the MAD

# Alternatively we can record the performance on the "validation set" during training

num_epochs <- 300
all_mae_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(partial_train_data, partial_train_targets, 
                  validation_data = list(val_data, val_targets),  
                  epochs = num_epochs, batch_size = 16, verbose = 0)
  
  mae_history <- history$metrics$val_mae
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

# avergaging the histories
average_mae_history <- data.frame(
  epoch = 1:ncol(all_mae_histories),
  validation_mae = apply(all_mae_histories, 2, mean)
)

# ploting
# the averages
plot(average_mae_history[-(1:10),],type="l",ylim = c(2,5))
# the individual histories
matlines(x=11:num_epochs, t(all_mae_histories[,-(1:10)]),col = 2:6,lty=1)

# using smoothing in ggplot
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

# This is different from what the book is showing
# I used batch = 16, instead of 1, which seemed to reduce overfitting

# After tuning the parameters, including number of epochs, we train the model using all training data

# for example
model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 90, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)
result

# Mean Absolute Deviation (MAD) from the data:
mean(abs(test_targets-mean(test_targets)))
# about 37%