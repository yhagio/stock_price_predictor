# Machine Learning Engineer Nanodegree
## Capstone Proposal
Yuichi Hagio  
July 22, 2017

## Proposal

Stock Price Predictor

### Domain Background

Machine learning can be great at predicting the future from the historical data. 
A machine-learning algorithm can be more accurate on the prediction 
than conventional trading strategies based on rules set by humans.

Investment firms have adopted machine learning in recent years rapidly, 
even some firms have started replacing humans with A.I. to make investment decisions.

References:

- http://fortune.com/2017/03/30/blackrock-robots-layoffs-artificial-intelligence-ai-hedge-fund/
- https://seekingalpha.com/article/4083754-superior-portfolio-roi-artificially-intelligent-algorithms

### Problem Statement

There is no easy way to predict stock prices accurately 
and no method is perfect since there are many factors
that can affect the stock prices (i.e. people's emotion, natural disasters, etc), 
but I believe that we can predict whether the closing price goes up or down by applying machine learning techniques and algorithm from the historical data set for this project. 

### Datasets and Inputs

There are several data sources for the historical stock price data.
I can use yahoo finance data set (`.csv` format) for this project.

The data includes following properties:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

How different is adjusted close price from close price?

Adjusted close price is the price of the stock at the closing of the trading adjusted with the dividends, and the close price is the price of the stock at the closing of the trading. Both values can be same, or not.

Data source links: (Daily hisorical prices - period: Jul 22, 2012 - Jul 22, 2017)

Download the CSV file for each (GE, S&P 500, Nikkei 225, Apple, Toyota)

- GE: https://finance.yahoo.com/quote/GE/history?p=GE
- S&P 500: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
- Nikkei 225: https://finance.yahoo.com/quote/%5EN225/history?p=%5EN225
- Apple: https://finance.yahoo.com/quote/AAPL/history?p=AAPL
- Toyota: https://finance.yahoo.com/quote/TM/history?p=TM

### Solution Statement

From the data set of yahoo finance. Predict the closing price of a target day based on the 
historical data up to the previous day. 

References:
- Machine Learning for Trading: https://www.udacity.com/course/machine-learning-for-trading--ud501
- Time Series Forecasting https://www.udacity.com/course/time-series-forecasting--ud980

### Benchmark Model




### Evaluation Metrics

To determine how accurate the prediction is, we analyze the difference between 
the predicted and the actual adjusted close price. Smaller the difference indicates better 
accuracy.

### Project Design

Use `numpy`, `pandas`, `sklearn`, `matplotlib`

Load Data from CSV file
Train the model
Test the model
Improve / Tune up
Result
Conclusion
