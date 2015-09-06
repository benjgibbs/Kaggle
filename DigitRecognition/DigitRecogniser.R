library(tree)
library(randomForest)
library(gbm)

baseDir <- "C:/Users/Ben/Documents/GitHub/Kaggle/DigitRecognition"
#baseDir <- "~/Code/Git/Kaggle/DigitRecognition"

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

strToDigit <- Vectorize(function(d)  {
    res <- NA
    if(d == "zero") res <- 0
    else if(d == "one") res <- 1
    else if(d == "two") res <- 2
    else if(d == "three") res <- 3
    else if(d == "four") res <- 4
    else if(d == "five") res <- 5
    else if(d == "six") res <- 6
    else if(d == "seven") res <- 7
    else if(d == "eight") res <- 8
    else if(d == "nine") res <- 9
    return (res)
})

loadTrainingData <-function() {
    digits <- read.csv(paste(baseDir,"train.csv", sep="/"))
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
  #0.937380952380952 with mtry=3 and nodesize=5
  #0.95797619047619 with mtry=6 and nodesize=10
    rf1 <- randomForest(Classifier~.-label, data=digits, subset=train,importance=TRUE, mtry=12, nodesize=20)
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

test=read.csv(paste(baseDir,"test.csv", sep="/"))
test$label = 100
test.predict <- predict(predictor, test, type="class")
test.predict.digit <- strToDigit(test.predict)
head(test.predict.digit)

outFile <- paste(baseDir, "out.csv", sep="/")
write.table(test.predict.digit, col.names="ImageId,Label", file=outFile, quote=F,row.names=T,sep=",")
