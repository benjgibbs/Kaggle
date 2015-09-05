library(tree)
library(randomForest)
library(gbm)

digitToStr <- Vectorize(function(d)  {
    res <- "error"
    if(d == 0) res <- "zero"
    else if(d == 1) res <- "one"
    else if(d == 2) res <- "two"
    else if(d == 3) res <- "three"
    else if(d == 4) res <- "four"
    else if(d == 5) res <- "five"
    else if(d == 6) res <- "six"
    else if(d == 7) res <- "seven"
    else if(d == 8) res <- "eight"
    else if(d == 9) res <- "nine"
    return (res)
})

loadTrainingData <-function() {
    digits <- read.csv("~/Code/Git/Kaggle/DigitRecognition/train.csv")
    Classifier <- digitToStr(digits[,"label"])
    digits <- data.frame(Classifier,digits)
    return (digits)
}

createTrainingSet <- function(digits, wt) {
    train.sz=nrow(digits)*wt
    set.seed(17)
    train=sort(sample(1:nrow(digits),train.sz))
    return (train)
}

buildTreePredictor <- function(digits, train) {
    t1 <- tree(Classifier~.-label, digits, subset=train)
    prunings <- cv.tree(t1, FUN=prune.misclass)
    k <- prunings$size[which.min(prunings$dev)]
    t2 <- prune.tree(t1,k=k)
    
    return (t2)
}

buildRfPredictor <- function(digits, train) {
    rf1 <- randomForest(Classifier~.-label, data=digits, subset=train,importance=TRUE, mtry=3, nodesize=5)
    return(rf1)
}

buildBoostingPredictor <- function(digits, train) {
    b1 <- gbm(Classifier~.-label, data=digits[train,])
    return(b1)
}

checkError <- function(digits,train,predictor) {
    check=digits[-train,]
    digit.predict <- predict(predictor, check, type="class")
    print(table(digit.predict,check[,"Classifier"]))
    
    check.result<-check[,"Classifier"]
    error<-sum(check.result == digit.predict)/length(check.result)
    print(paste("Error is", error))
}

digits <- loadTrainingData()
train <- createTrainingSet(digits, 0.8)
#predictor <- buildTreePredictor(digits, train)
#predictor <- buildBoostingPredictor(digits, train)
predictor <- buildRfPredictor(digits, train)

checkError(digits,train,predictor)

#test=read.csv("~/Code/Git/Kaggle/DigitRecognition/test.csv")
#test.predict <- predict(tree.digits, test, type="class")
