library(tidyverse)


spam_email <- read.csv("C:/Users/USNHIT/Desktop/Machine Learning Projects/Spam Email Project/spambase_csv.csv")
head(spam_email)

#percentage of words in the e-mail that match WORD, i.e.
#100 * (number of times the WORD appears in the e-mail) 
#/ total number of words in e-mail. A “word” in this case
#is any string of alphanumeric characters bounded by 
#non-alphanumeric characters or end-of-string.

#Data cleaning 

#missing values
columns_with_missing_values <- spam_email%>%
  summarise(across(everything(),~sum(is.na(.)))) %>%
  pivot_longer(cols = everything(),names_to = "Column", values_to = "Missing_values")%>%
  filter(Missing_values>0)

#duplicates
duplicates <- spam_email%>%
  distinct()%>%
  filter(duplicated(spam_email))

nrow(duplicates)


#EDA ###############################################

#data distribution  
#1813 out of 4601 are spam emails  

spam_email%>%
  group_by(spam_email$class)%>%
  summarise(sum = sum(class))

#which set of words have highest proportion in spam email?


proportion  <- spam_email%>%
  filter(class == 1)%>%
  select(-capital_run_length_average,-capital_run_length_longest,-capital_run_length_total)

boxplot(proportion)

max_values <- apply(proportion,2,max, na.rm=TRUE) #1 for rows & 2 for columns

plot(max_values)  #3D words has highest frequency 
#(font, free, credit, .23) have higher values also. 


# test train split

size_email = round(nrow(spam_email)*0.22)

testing = sample(nrow(spam_email),size_email)

train_spam =  spam_email[-testing,]
test_spam = spam_email[testing,]


log_spam <- glm(class ~.,train_spam, family = binomial)
summary(log_spam)$coef

log_prob <- predict(log_spam,newdata = test_spam, type = "response")

log_pred <- rep(0, nrow(test_spam))
log_pred[log_prob>=0.5] = 1
log_pred = as.factor(log_pred)

#921 values out of 1012 were predicted correctly which is not bad.

library(caret)

table(log_pred, True = test_spam[,"class"])

round(mean(log_pred!=test_spam[,"class"]),4) # error rate is 7% 


cm = confusionMatrix(log_pred, as.factor(test_spam$class))
# Accuracy  = 93%

cm$byClass
#Precision = 93.2%
#Recall = 95.47%
# Balanced accuracy = 92.27%
# F1 = 94.23%

library(pROC)

roc.test = roc(test_spam$class~log_prob,plot = TRUE,print.auc = TRUE)

#0.971 

#cross validation 

cost<- function(class,pi=0) mean(abs(class-pi)>0.5)
library(boot)
set.seed(11)
cv.glm(train_spam,log_spam,cost, K=10)$delta


# Linear Discriminant Analysis

library(MASS)

lda_m = lda(class~.,data = train_spam)
lda_m

plot(lda_m)

lda.predict = predict(lda_m,newdata = test_spam)

table(test_spam$class,lda.predict$class)

lda.predict$class

spam_email = spam_email[,1:55]

lda_cv <- lda(class~.,CV=TRUE,data=spam_email, subset = -testing)

head(lda_cv$class)

# getting error 
# object is not matrix 


qda_m <- qda(class~.,data= train_spam)
qda_m

qda.pred = predict(qda_m,newdata= test_spam)

table(test_spam$class,qda.pred$class)

qda_cm = confusionMatrix(qda.pred$class, as.factor(test_spam$class))
#Accuracy = 0.834

qda_cm$byClass
#Recall = 75.24%
# Precision  = 96.8%
# F1 score = 84.69%
#Balanced Accuracy = 85.71%



# Data prep for KNN 
knn_train_spam = train_spam[,1:55]
knn_test_spam = train_spam[,1:55]
knn_train_labels = train_spam[,"class"]
knn_test_labels = test_spam[,"class"]

train_features <- train_spam[, -ncol(train_spam)]
train_labels <- train_spam[, ncol(train_spam)]
test_features <- test_spam[, -ncol(test_spam)]
test_labels <- test_spam[, ncol(test_spam)]

knn3 = knn(train = train_features, test = test_features, cl = train_labels, k=3)
library(class)
dim(knn3)
table(knn3, (test_labels))

1-mean(knn3 ==test_labels)
#0.1946 error rate 


knn5 = knn(train = train_features, test= test_features, cl = train_labels, k =7)

table(knn5,test_labels)
1-mean(knn5 ==test_labels)
# for knn =7, error rate =0.18972


# choosing best k value for knn

set.seed(122)
k.grid= 1:100
error = rep(0, length(k.grid))

for (i in seq_along(k.grid)) {
  pred = knn(train = scale(train_features), 
             test = scale(test_features),
             cl = train_labels,
             k = k.grid[i])
  error[i] = mean(test_labels!= pred)
  
}

min(error)

plot(k.grid,error)
#Why error is increasing as K value increases?

#As K increases, the model's complexity decreases, due to that it becomes less sensitive to the training dataset.
# Overall, it has low varaince and high bais which leads to under fitting

