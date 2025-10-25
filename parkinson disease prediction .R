# ============================================
# PARKINSON'S DISEASE PREDICTION IN R
# ============================================

# Load required libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(e1071)
library(randomForest)
library(class)
library(rpart)
library(rpart.plot)
library(GGally)
library(corrplot)

# ============================================
# 1. Load Dataset
# ============================================
# You can download the dataset from UCI ML Repository:
# https://archive.ics.uci.edu/ml/datasets/parkinsons
# Assume file = "parkinsons.data" 
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
download.file(url, destfile = "parkinsons.data", method = "auto")

data <- read.csv("parkinsons.data")
data <- read.csv("parkinsons.data")

# Remove name column (not useful)
data <- data[ , -1]

# Check structure
str(data)

# ============================================
# 2. Exploratory Data Analysis (EDA)
# ============================================
summary(data)

# Correlation matrix
corrplot(cor(data[,-ncol(data)]), method = "color", tl.cex = 0.6)

# Distribution of target variable
ggplot(data, aes(x = factor(status), fill = factor(status))) +
  geom_bar() +
  scale_fill_manual(values = c("#00BFC4", "#F8766D"),
                    labels = c("Healthy", "Parkinson’s")) +
  labs(title = "Parkinson’s Status Distribution", x = "Status", y = "Count")

# Pair plot for top features
GGally::ggpairs(
  data[, c("MDVP.Fo.Hz.", "MDVP.Fhi.Hz.", "MDVP.Flo.Hz.",
           "MDVP.Jitter...", "MDVP.Shimmer", "status")],
  aes(color = factor(status))
)

# ============================================
# 3. Data Preprocessing
# ============================================
set.seed(123)
data$status <- factor(data$status, levels = c(0,1), labels = c("Healthy", "Parkinson"))

# Split train-test
trainIndex <- createDataPartition(data$status, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]

# ============================================
# 4. Model Training
# ============================================

# ---- Logistic Regression ----
model_lr <- train(status ~ ., data = trainData, method = "glm", family = "binomial")

# ---- Decision Tree ----
model_dt <- rpart(status ~ ., data = trainData, method = "class")

# ---- Random Forest ----
model_rf <- randomForest(status ~ ., data = trainData, ntree = 100)

# ---- SVM ----
model_svm <- svm(status ~ ., data = trainData, kernel = "radial")

# ---- KNN ----
control <- trainControl(method="cv", number=5)
model_knn <- train(status ~ ., data=trainData, method="knn", trControl=control)

# ============================================
# 5. Predictions
# ============================================
pred_lr  <- predict(model_lr,  testData)
pred_dt  <- predict(model_dt,  testData, type="class")
pred_rf  <- predict(model_rf,  testData)
pred_svm <- predict(model_svm, testData)
pred_knn <- predict(model_knn, testData)

# ============================================
# 6. Accuracy Evaluation
# ============================================
acc_lr  <- mean(pred_lr  == testData$status)
acc_dt  <- mean(pred_dt  == testData$status)
acc_rf  <- mean(pred_rf  == testData$status)
acc_svm <- mean(pred_svm == testData$status)
acc_knn <- mean(pred_knn == testData$status)

accuracy_df <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "kNN"),
  Accuracy = c(acc_lr, acc_dt, acc_rf, acc_svm, acc_knn)
)

# ============================================
# 7. Visualization of Accuracy
# ============================================
ggplot(accuracy_df, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(aes(label = round(Accuracy,3)), hjust = 1.1, color = "white", size = 4) +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

# ============================================
# 8. Decision Tree Visualization
# ============================================
rpart.plot(model_dt, main="Decision Tree for Parkinson’s Prediction")

# ============================================
# 9. Feature Importance (Random Forest)
# ============================================
varImpPlot(model_rf, main="Feature Importance - Random Forest")

# ============================================
# END
# ============================================
set.seed(123)  # ensures reproducible results

# Convert status to factor
data$status <- factor(data$status, levels = c(0,1), labels = c("Healthy","Parkinson"))

# Split the data: 80% train, 20% test
trainIndex <- createDataPartition(data$status, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]  # training set
testData  <- data[-trainIndex, ] # test set

# ---- Logistic Regression ----
model_lr <- train(status ~ ., data = trainData, method = "glm", family = "binomial")

# ---- Random Forest ----
model_rf <- train(status ~ ., data = trainData, method = "rf")

# ---- SVM ----
model_svm <- train(status ~ ., data = trainData, method = "svmRadial")

# ---- K-Nearest Neighbors ----
model_knn <- train(status ~ ., data = trainData, method = "knn")

# ---- Decision Tree ----
model_dt <- train(status ~ ., data = trainData, method = "rpart")

library(kernlab)
model_svm <- train(status ~ ., data = trainData, method = "svmRadial")

# K-Nearest Neighbors
model_knn <- train(status ~ ., data = trainData, method = "knn")

# Decision Tree
model_dt <- train(status ~ ., data = trainData, method = "rpart")

# Logistic Regression
pred_lr <- predict(model_lr, newdata = testData)
acc_lr <- mean(pred_lr == testData$status)

# Random Forest
pred_rf <- predict(model_rf, newdata = testData)
acc_rf <- mean(pred_rf == testData$status)

# SVM
pred_svm <- predict(model_svm, newdata = testData)
acc_svm <- mean(pred_svm == testData$status)

# KNN
pred_knn <- predict(model_knn, newdata = testData)
acc_knn <- mean(pred_knn == testData$status)

# Decision Tree
pred_dt <- predict(model_dt, newdata = testData)
acc_dt <- mean(pred_dt == testData$status) 

accuracy_df <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree"),
  Accuracy = c(acc_lr, acc_rf, acc_svm, acc_knn, acc_dt)
)
library(ggplot2)

ggplot(accuracy_df, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(aes(label = round(Accuracy, 3)), hjust = 1.1, color = "white", size = 4) +
  labs(title = "Parkinson's Disease Prediction - Model Accuracy", x = "Model", y = "Accuracy") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")  

library(corrplot)

# Compute correlation matrix (exclude status column)
cor_matrix <- cor(data[ , sapply(data, is.numeric)])

# Plot correlation heatmap
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = "orange", addCoef.col = "black")

# Shimmer boxplot
ggplot(data, aes(x = status, y = MDVP.Shimmer, fill = status)) +
  geom_boxplot() +
  labs(title = "Shimmer Distribution by Status", x = "Status", y = "MDVP Shimmer") +
  theme_minimal()

# Jitter boxplot
ggplot(data, aes(x = status, y = MDVP.Jitter..., fill = status)) +
  geom_boxplot() +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "Jitter Distribution by Status", x = "Status", y = "MDVP Jitter") +
  theme_minimal()
ggplot(data, aes(x = MDVP.Fo.Hz., y = MDVP.Shimmer, color = status)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "MDVP Fo vs MDVP Shimmer", x = "MDVP Fo (Hz)", y = "MDVP Shimmer") +
  theme_minimal()

ggplot(data, aes(x = MDVP.Fo.Hz., fill = status)) +
  geom_histogram(alpha = 0.6, bins = 20, position = "identity") +
  labs(title = "Distribution of MDVP Fo", x = "MDVP Fo (Hz)", y = "Count") +
  theme_minimal()

