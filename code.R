# necessay imports of library
library(dplyr)
library(tidyr)
library(lubridate)
library(DataCombine)
library(glmnet)
library(broom)
library(ggplot2)
options(tibble.print_max = Inf)

# set up working directory
setwd("~/Downloads/MSBA/2020 Spring/MS ML/HW/data1")

#### Question 1 ####
# read dataset
# DJ is the dataset downloaded from Wharton CRSP
DJ <- read.csv("DJIA.csv", stringsAsFactors = FALSE)
DJ <- transform(DJ, date = as.Date(as.character(date), "%Y%m%d"))
DJ <- transform(DJ, RET = as.double(RET))
summary(DJ)

# code is the list of Ticker Codes obtained from wikipeida
code <- read.csv("code.txt", header = FALSE, stringsAsFactors = FALSE)

# groupby TICKER to examine data quality
DJ %>% group_by(TICKER) %>% summarise(N = n())

# examine stocks with missing values/dates
tickers <- unique(DJ$TICKER)
 
setdiff(tickers, unique(code$V1)) # ARNC, HWP, CMB, SBC

print(summary(DJ[DJ$TICKER=="ARNC",])) # was AA before Nov 2016
print(summary(DJ[DJ$TICKER=="AA",])) # AA has more comprehensive data, drop ARNC

print(summary(DJ[DJ$TICKER=="HWP",])) # HWP changed to HPQ after 2002/05/06
print(summary(DJ[DJ$TICKER=="HPQ",])) # merge these two into one column HPQ

print(summary(DJ[DJ$TICKER=="CMB",])) # could not find CMB related stock info
print(tail(DJ[DJ$PERMNO==47896,])) # JPM has the same PERMNO number with CMB, drop CMB

print(summary(DJ[DJ$TICKER=="SBC",])) # SBC changed to T after 2005
print(summary(DJ[DJ$TICKER=="T",])) # T has more comprehensive data, drop SBC

print(summary(DJ[DJ$TICKER=="GM",])) # GM has records across whole time span, keep it
print(summary(DJ[DJ$TICKER=="KODK",])) # has records only after 2013, drop it

DJ_2 <- DJ
DJ_2$TICKER[DJ_2$TICKER=="HWP"] <- "HPQ"
DJ_2 <- DJ_2[DJ_2$TICKER!="ARNC",]
DJ_2 <- DJ_2[DJ_2$TICKER!="CMB",]
DJ_2 <- DJ_2[DJ_2$TICKER!="SBC",]
DJ_2 <- DJ_2[DJ_2$TICKER!="KODK",]

length(unique(DJ_2$TICKER)) # 28

# chage data format, put each stock into columns
DJ_3 <- DJ_2[,2:4] %>% 
  pivot_wider(names_from = TICKER,
              values_from = RET,
              values_fn = list(RET = mean))

summary(DJ_3) # 4279 records of 29 variables (date + 28 stocks)

#### Question 2 (a) ####
# split training set 1
train1 <- DJ_3 %>% 
  select(colnames(DJ_3)) %>% 
  filter(between(date, 
                 as.Date("2000-1-1"), 
                 as.Date("2005-12-31")))
print(summary(train1))

# split training set 2
train2 <- DJ_3 %>% 
  select(colnames(DJ_3)) %>% 
  filter(between(date, 
                 as.Date("2006-01-01"), 
                 as.Date("2010-12-31")))
print(summary(train2))

# split test set
test <- DJ_3 %>% 
  select(colnames(DJ_3)) %>% 
  filter(between(date, 
                 as.Date("2011-01-01"), 
                 as.Date("2016-12-31")))
print(summary(test))

# PCA on train1
train1 <- na.omit(train1)
pr.out <- prcomp(train1[,2:29], scale=TRUE)
ratio <- pr.out$rotatio
pr.var <- pr.out$sdev^2
pve <- pr.var/sum(pr.var)
print(pve)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained ", ylim=c(0,1),type="b")
plot(cumsum(pve), xlab="Principal Component ", ylab=" Cumulative Proportion of Variance Explained ", ylim=c(0,1), type="b")

#### Question 2 (b) ####
# apply principal component loadings on train2
# replace NA in GM with 0
train2_fill <- train2 %>% replace_na(list(GM = 0))
train2_pc <- as.matrix(train2_fill[,2:29]) %*% as.matrix(ratio)
train2_pc <- as.data.frame(train2_pc)
train2_pc <- cbind(date = train2_fill$date,train2_pc)

# take the top 5 components and lag from -1 to -5
slide_df <- train2_pc[,1:6]
for (pc in setdiff(names(slide_df),"date")){
  for (i in -5:-1){
    slide_df <- slide(slide_df,TimeVar = 'date', Var = pc, slideBy = i)
  }
}
head(slide_df)

# combine lagged 5 components with 1st stock (lead 1 term)
RET_df <- as.data.frame(train2_fill[,1:2])
RET_df <- slide(RET_df,TimeVar = 'date', Var = "MSFT", NewVar = "RET", slideBy = 1)
RET_df <- RET_df[,c(1,3)]
lagged_df <- merge(RET_df, slide_df, by="date", all=TRUE)
lagged_df$ticker="MSFT"

#### Question 2 (c) ####
for (tic in setdiff(names(train2_fill),c("date","MSFT"))){
  RET_df <- as.data.frame(train2_fill[,c("date",tic)])
  RET_df <- slide(RET_df,TimeVar = 'date', Var = tic, NewVar = "RET", slideBy = 1)
  RET_df <- RET_df[,c(1,3)]
  stock_df <- merge(RET_df, slide_df, by="date", all=TRUE)
  stock_df$ticker=tic
  lagged_df <- rbind(lagged_df,stock_df)
}
lagged_df$ticker <- factor(lagged_df$ticker)
summary(lagged_df)
View(lagged_df[,c(1:3,8:12,4)])

#### Question 3 (a) ####
# omit NAs
full_lag_df <- na.omit(lagged_df)

# set interaction terms
tickers <- full_lag_df$ticker
f <- as.formula( ~ .* tickers)
x <- model.matrix(f, full_lag_df[,3:32])[, -1]
dim(x)

# set y values
y <- full_lag_df$RET
length(y)

# fit LASSO models
lasso.mod <- glmnet(x, y, alpha = 1)
plot(lasso.mod)

#### Question 3 (b) ####
# fit LASSO models with 5-fold cross validation
set.seed(1)
cv.out <- cv.glmnet(x, y, alpha = 1, nfolds = 5)

# plot of lambda against MSE
plot(cv.out)

# choose the best lambda
bestlam <- cv.out$lambda.min
bestlam

# fit LASSO model with best lambda on train2
lasso.mod.cv <- glmnet(x, y, alpha = 1, lambda = bestlam)
tidy(coef(lasso.mod.cv))
train.pred <- predict(lasso.mod.cv, s = bestlam, newx = x)
MSE.train <- mean((train.pred - y)^2)
MSE.train


#### Question 4 ####
# construct PC for test set
test_fill <- test %>% replace_na(list(AA=0))
test_pc <- as.matrix(test_fill[,2:29]) %*% as.matrix(ratio)
test_pc <- as.data.frame(test_pc)
test_pc <- cbind(date = test_fill$date,test_pc)
test.slide_df <- test_pc[,1:6]

for (pc in setdiff(names(test.slide_df),"date")){
  for (i in -5:-1){
    test.slide_df <- slide(test.slide_df,TimeVar = 'date', Var = pc, slideBy = i)
  }
}
head(test.slide_df)

# fill the NA with 0.0
test.slide_df[is.na(test.slide_df)] <- 0.0

# create the lagged dataframe
test.RET_df <- as.data.frame(test_fill[,1:2])
test.RET_df <- slide(test.RET_df,TimeVar = 'date', Var = "MSFT", NewVar = "RET", slideBy = 1)
test.RET_df <- test.RET_df[,c(1,3)]
test.lagged_df <- merge(test.RET_df, test.slide_df, by="date", all=TRUE)
test.lagged_df$ticker="MSFT"

for (tic in setdiff(names(DJ_3),c("date","MSFT"))){
  test.RET_df <- as.data.frame(test_fill[,c("date",tic)])
  test.RET_df <- slide(test.RET_df,TimeVar = 'date', Var = tic, NewVar = "RET", slideBy = 1)
  test.RET_df <- test.RET_df[,c(1,3)]
  test.stock_df <- merge(test.RET_df, test.slide_df, by="date", all=TRUE)
  test.stock_df$ticker=tic
  test.lagged_df <- rbind(test.lagged_df,test.stock_df)
}
test.lagged_df$ticker <- factor(test.lagged_df$ticker)
test.lagged_df <- na.omit(test.lagged_df)

# set x values
tickers <- test.lagged_df$ticker
test.f <- as.formula( ~ .*tickers)
test.x <- model.matrix(test.f, test.lagged_df[,3:32])[, -1]
dim(test.x)

# set y values
test.y <- test.lagged_df$RET
length(test.y)

# predict values
lasso.pred <- predict(lasso.mod.cv, s = bestlam, newx = test.x)
MSE.test <- mean((lasso.pred - test.y)^2)
MSE.test
# attach the values with original RET data
fac_ticker <- as.character(test.lagged_df$ticker)
pred.result <- as.data.frame(test.lagged_df$date)
names(pred.result) <- "DATE"
pred.result$RET <- test.lagged_df$RET
pred.result$TICKER <- fac_ticker
pred.result$PREDICT <- as.double(lasso.pred)
summary(pred.result)
head(pred.result,n=10)

f <- function(x){ if (x) 1 else -1}

#### trading strategy 1 ####
capital1 <- c(100)

for (test.day in unique(pred.result$DATE)){
  day_df <- pred.result[pred.result$DATE==test.day,]
  ord_df <- day_df[order(day_df$PREDICT),]
  
  # for a given date, find the top 5 and bottom 5 predictions
  # in this strategy, we go long for those with positive predictions
  # and short those with negative predictions
  select_stocks <- rbind(ord_df[1:5,],ord_df[24:28,])
  
  # examine whether the prediction and reality are of the same direction
  select_stocks$sign <- select_stocks$RET * select_stocks$PREDICT > 0
  # for the same direction, we get positive return
  select_stocks$sign <- as.numeric(lapply(select_stocks$sign,f))
  
  # calculate the return based on true RET
  select_stocks$return <- abs(select_stocks$RET) * select_stocks$sign / 10
  
  # sum up the total return and add to total capital
  total_return <- sum(select_stocks$return)
  capital1.old <- tail(capital1,1)
  capital1.new <- capital1.old * (1 + total_return)
  capital1 <- c(capital1,capital1.new)
}
tail(capital1)
plot(unique(pred.result$DATE), capital1[2:1510], xlab="Date", ylab="Total Capital", type = "l")
best.total1 <- max(capital1)
best.day1 <- match(capital1[capital1==best.total1],capital1)
best.date1 <- unique(pred.result$DATE)[best.day1]
print(list(best.total1,best.date1))
return_rate1 <- (tail(capital1,1)/capital1[1])^(1/7)
return_rate1

#### trading strategy 2 ####
capital2 <- c(100)

for (test.day in unique(pred.result$DATE)){
  day_df <- pred.result[pred.result$DATE==test.day,]
  ord_df <- day_df[order(day_df$PREDICT),]
  
  #  for a given date, find the top 5 and bottom 5 predictions
  select_stocks <- rbind(ord_df[1:5,],ord_df[24:28,])
  
  # in this strategy, we go short the bottom 5 stocks
  select_stocks$sign <- -1
  # and long the top 5 stocks
  select_stocks$sign[6:10] <- 1
  
  # calculate the return based on true RET
  select_stocks$return <- select_stocks$RET * select_stocks$sign / 10
  
  # sum up the total return and add to total capital
  total_return <- sum(select_stocks$return)
  capital2.old <- tail(capital2,1)
  capital2.new <- capital2.old * (1 + total_return)
  capital2 <- c(capital2,capital2.new)
}
tail(capital2)
plot(unique(pred.result$DATE), capital2[2:1510], xlab="Date", ylab="Total Capital", type = "l")
best.total2 <- max(capital2)
best.day2 <- match(capital2[capital2==best.total2],capital2)
best.date2 <- unique(pred.result$DATE)[best.day2]
print(list(best.total2,best.date2))
return_rate2 <- (tail(capital2,1)/capital2[1])^(1/7)
return_rate2

# compare the resutls of two strategies
c1 <- data.frame(Date=unique(pred.result$DATE), Total_Capital=capital1[2:1510])
c2 <- data.frame(Date=unique(pred.result$DATE), Total_Capital=capital2[2:1510])
combined_capital <- c1 %>% mutate(Strategy = 'S1') %>% bind_rows(c2 %>% mutate(Strategy = 'S2'))

ggplot(combined_capital,
       aes(y = Total_Capital,x = Date,color = Strategy)) + geom_line() + theme_bw()

