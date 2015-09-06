library(tree)
library(randomForest)
library(gbm)

baseDir <- "C:/Users/Ben/Documents/GitHub/Kaggle/Titanic"
#baseDir <- "~/Code/Git/Kaggle/Titanic"

survivedToStr <- Vectorize(function(x) {
  return (ifelse(x, "Survived", "Died"))
})

strToSurvived <- Vectorize(function(x) {
  return (ifelse(x=="Survived", 1, 0))
})

loadTrainingData <-function() {
  titanic <- read.csv(paste(baseDir,"train.csv", sep="/"))
  SurvivedStr <- survivedToStr(titanic[,"Survived"])
  titanic <- data.frame(SurvivedStr,titanic)
  return (titanic)
}

cleanData <- function(titanic) {
  
  maleAge <- mean(titanic[titanic$Sex == "male" & !is.na(titanic$Age) ,]$Age)
  femaleAge <- mean(titanic[titanic$Sex == "female" & !is.na(titanic$Age) ,]$Age)
  CleanAge <- ifelse(is.na(titanic$Age), ifelse(titanic$Sex == "male",maleAge,femaleAge), titanic$Age)
  thirdClassFare <- mean(titanic[titanic$Pclass == 3 & !is.na(titanic$Fare) ,]$Fare)
  Fare <- ifelse(is.na(titanic$Fare),thirdClassFare,titanic$Fare)
  titanic <- data.frame(SurvivedStr = titanic$SurvivedStr,
                        PassengerId = titanic$PassengerId,
                        Sex = titanic$Sex,
                        CleanAge = CleanAge,
                        SibSp = titanic$SibSp,
                        Parch = titanic$Parch, 
                        Fare = Fare, 
                        Pclass = titanic$Pclass, 
                        Embarked = titanic$Embarked)
  return(titanic)
}

createTrainingSet <- function(titanic, wt) {
  train.sz=nrow(titanic)*wt
  set.seed(173)
  train=sort(sample(1:nrow(titanic),train.sz))
  return (train)
}

buildTreePredictor <- function(titanic, train) {
  #Age+Pclass+Sex+SibSp+Parch+Fare+Embarked
  t1 <- tree(SurvivedStr~Sex+Age+Pclass+Sex+SibSp+Parch+Fare+Embarked, titanic, subset=train)
  prunings <- cv.tree(t1, FUN=prune.misclass)
  k <- prunings$size[which.min(prunings$dev)]
  t2 <- prune.tree(t1,k=k)
  
  return (t2)
}

buildRfPredictor <- function(titanic, train) {
  #Age+Pclass+Sex+SibSp+Parch+Fare+Embarked
  rf1 <- randomForest(SurvivedStr~Sex+Pclass+Sex+SibSp+Parch+Fare+Embarked+CleanAge, 
                      data=titanic, subset=train,importance=TRUE)
  return (rf1)
}

checkError <- function(titanic,train,predictor) {
  check=titanic[-train,]
  survived.predict <- predict(predictor, check, type="class")
  print(table(survived.predict,check[,"SurvivedStr"]))
  
  check.result<-check[,"SurvivedStr"]
  error<-sum(check.result == survived.predict)/length(check.result)
  print(paste("Error is", error))
}


titanic <- loadTrainingData()
titanic <- cleanData(titanic)
train <- createTrainingSet(titanic, 0.90)
#predictor <- buildTreePredictor(titanic, train)
predictor <- buildRfPredictor(titanic, train)
checkError(titanic,train,predictor)

test=read.csv(paste(baseDir,"test.csv", sep="/"))
test$SurvivedStr <- ""
test <- cleanData(test)
levels(test$Embarked) <- c("", "C", "Q", "S")
test.predict <- predict(predictor, test, type="class")
test.predict.survived <- strToSurvived(test.predict)

out <- data.frame(PassengerId = test$PassengerId, Survived = as.integer(test.predict.survived))


outFile <- paste(baseDir, "out.csv", sep="/")
write.table(out, file=outFile, quote=F,row.names=F,sep=",")


