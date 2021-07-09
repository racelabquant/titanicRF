library(aod)
library(ggplot2)
library(dbplyr)

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

#test1 <- with(mydata, data.frame(gre = mean(gre), gpa = mean(gpa), rank = factor(1:4)))
## view data frame
#newdata1

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

# Random Forest
library(tidymodels)
# specify model settings
decision_tree() %>% set_engine("rpart") %>% set_mode("classification")

# the specification is a skeleton and needs to be trained with data
tree_spec <- decision_tree() %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tree_spec %>% fit(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=mydata)
