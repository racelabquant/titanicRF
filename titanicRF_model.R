library(aod)
library(ggplot2)
library(dbplyr)
library(randomForest)
library(rpart)
library(rpart.plot)
library(pander)
library(caTools)
library(rattle)
library(RColorBrewer)

mydata <- read.csv("G:/My Drive/Research Clients/RaceLab/Data/Testing/titanic_munged.csv")
## view the first few rows of the data
head(mydata)
summary(mydata)

# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

sapply(mydata, sd)

str(mydata)

# Factorise the data
mydata$Sex <- factor(mydata$Sex)
mydata$Survived <- factor(mydata$Survived)
mydata$Pclass <- factor(mydata$Pclass)
mydata$Embarked <- factor(mydata$Embarked)

k <- ggplot(mydata, aes(Survived))
k + geom_bar(aes( fill = Sex), width=.85, colour="darkgreen") + scale_fill_brewer() +
  ylab("Survival Count (Genderwise)") +
  xlab("Survived: No = 0, Yes = 1") +
  ggtitle("Titanic Disaster: Gender Vs. Survival (Training Dataset")

# first logit
mylogit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = mydata, family = "binomial")
summary(mylogit)
# second logit
mylogit <- glm(Survived ~ Pclass + Sex + Age + SibSp, data = mydata, family = "binomial")
summary(mylogit)

## CIs using profiled log-likelihood
confint(mylogit)

## CIs using standard errors
confint.default(mylogit)

wald.test(b = coef(mylogit), Sigma = vcov(mylogit), Terms = 2:6)

## odds ratios only
exp(coef(mylogit))

## odds ratios and 95% CI
exp(cbind(OR = coef(mylogit), confint(mylogit)))

# prediction model (on original data set to test)
pred.titanic <- predict(mylogit, newdata = mydata, type = "response")

pred.titanic <- ifelse(pred.titanic > 0.5, 1, 0)
head(pred.titanic)

# Model accuracy
mean(pred.titanic == mydata$Survived)
# should be 78.79% prediction if using the same data set

# another approach to the prediction process
probabilities <- mylogit %>% predict(mydata, type = "response")

predicted.classes <- ifelse(probabilities > 0.5, 1, 0)

# Model accuracy
mean(predicted.classes == mydata$Survived)

### DECISION TREE ######
formula <- Survived ~ Sex + Pclass + Age

# Split the training data set so there is a test set
train_split <- initial_split(mydata,prop=0.8)
train_split # shows the split numbers
t_train <- training(train_split)
t_test <- testing(train_split)
nrow(t_train)/nrow(mydata) # verify % split

# Build the decision tree
dtree <- rpart(formula, data=t_train, method="class")

# Measure Performance on the Training Data
dtree_tr_predict <- predict(dtree, newdata=t_train, type="class")
dtree_tr_predict.t <- table(t_train$Survived, dtree_tr_predict)
# Model Accuracy
dtree_tr_accuracy <- (dtree_tr_predict.t[1, 1] + dtree_tr_predict.t[2, 2]) / sum(dtree_tr_predict.t)
# Print accuracy in Prediction
cat("Model Accuracy on Sub sample on training data: ", dtree_tr_accuracy)

# Measure Performance on the Test Data
dtree_te_predict <- predict(dtree, newdata=t_test, type="class")
dtree_te_predict.t <- table(t_test$Survived, dtree_te_predict)
# Model Accuracy
dtree_testing_accuracy <- (dtree_te_predict.t[1, 1] + dtree_te_predict.t[2, 2]) / sum(dtree_te_predict.t)
# Print accuracy
cat("Model Accuracy in Prediction: ", dtree_testing_accuracy)

# Plot the Decision Tree
fancyRpartPlot(dtree)


### RANDOM FOREST ######







# Other forms of decision tree
library(tidymodels)
# specify model settings
decision_tree() %>% set_engine("rpart") %>% set_mode("classification")

# the specification is a skeleton and needs to be trained with data
tree_spec <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tree_spec %>% fit(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=mydata)
tree_spec %>% fit(formula = Survived ~ Pclass + Sex + Age + SibSp, data=mydata)



# splits the data automatically
titanic_split <- initial_split(mydata,prop=0.8)
titanic_split # shows the split numbers
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)
nrow(titanic_train)/nrow(mydata) # verify % split

# check to avoid class imbalances
counts_train <- table(titanic_train$Survived)
counts_train
prop_surv_train <- counts_train["1"]/sum(counts_train)
prop_surv_train
counts_test <- table(titanic_test$Survived)
counts_test
prop_surv_test <- counts_test["1"]/sum(counts_test)
prop_surv_test
# if the proportions don't match then re-run the initial_split function
# with a strata argument initial_split(mydata, prop=0.8, strata=outcome)
example <- initial_split(mydata, prop=0.8, strata=Survived)
example

##### Predicting new data
# Arguments (1. Trained Model, 2. data set to predict on, 3. prediction type
# labels or probabilities)
predict(tree_spec, new_data=titanic_test, type = "class")


#### Create the confusion matrix

# combine predictions and truth values
pred_combined <- predictions %>%
  mutate(true_class = test_data$outcome)

pred_combined

# calculate the confusion matrix..three arguments needed for conf_mat
# conf_mat(data, estimate, truth)
conf_mat(data=pred_combined,
         estimate = .pred_class,
         truth = true_class)

# call yardstick library to automatically compute confusion matrices
# accuracy(data, estimate, truth) - it outputs a tibble
library(yardstick)
accuracy(pred_combined,
         estimate = .pred_class,
         truth = true_class)


