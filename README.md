# Lasso model

This is a practice of Lasso model in predicting stock returns.

Stocks selected are components of Dow Jones Industrial Average in January 2000.

The data we used is downloaded from Wharton Research Data Servies (WRDS), from Jan 1st 2000 to Dec 31st 2016.

In this model, we used top 5 principal components of daily stock returns and lagged from T-1 to T-5, as inputs to predict T+1 returns. Dummy variables for each stock are introduced, as well as the interaction term between dummies and PCs.

In this practice, we used data from Jan 1st 2000 to Dec 31st 2005 as training set 1 for calculating PCs.

We used data from Jan 1st 2006 to Dec 31st 2010 as training set 2 to select the Lasso model with 5-fold cross validation.

We used data from Jan 1st 2011 to Dec 31st 2016 as test set for calculating expected returns using this model.

We compared results of two different trading strategies in the test set.

Strategy 1 seems to have higher variance but also higher returns according to the resutls.
