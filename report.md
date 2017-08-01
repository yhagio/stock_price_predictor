# Machine Learning Engineer Nanodegree
## Capstone Project
Yuichi Hagio
July 30, 2017

## I. Definition

Stock Price Predictor

### Project Overview

Machine learning can be great at predicting the future from the historical data. 
A machine-learning algorithm can be more accurate on the prediction 
than conventional trading strategies based on rules set by humans.

Investment firms have adopted machine learning in recent years rapidly, 
even some firms have started replacing humans with A.I. to make investment decisions.

In this project, simply I experimented to use Deep Learning to predict stock prices.

### Problem Statement

There is no easy way to predict stock prices accurately 
and no method is perfect since there are many factors
that can affect the stock prices (i.e. people's emotion, natural disasters, etc), 
but I believe that I can predict whether the closing price goes up or down by applying machine learning techniques and algorithm from the historical data set for this project. 


### Metrics

To determine how accurate the prediction is, we analyze the difference between 
the predicted and the actual adjusted close price. Smaller the difference indicates better 
accuracy. 

I chose both Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) 
as a metric to determine the accuracy of the prediction. 
It is a commonly used general purpose quality estimator.

Also, by visualizing the predicted price and the actual price with a plot or a graph, it can tell how close the 
prediction is clearly.


**Why I use MSE/RMSE for the metric?**

There are many metrics for accuracy like R2, MAE, etc.

I chose to use MSE/RMSE because they explicitly show the deviation of the prediction for continuous variables
from the actual dataset. So, they fit in this project to measure the accuracy.

![alt text](images/rmse.gif "RMSE")

It measures the average magnitude of the error and ranges from 0 to infinity.
The errors are squared and then they are averaged,
MSE/RMSE gives a relatively high weight to large errors, and the errors in stock price prediction
can be critical, so it is appropriate metric to penalize the large errors.


## II. Analysis

### Data Exploration

There are several data sources for the historical stock price data.
I can use yahoo finance data set (`.csv` format) for this project.

**_How different is adjusted close price from close price?_**

Adjusted close price is the price of the stock at the closing of the trading adjusted with the dividends, and the close price is the price of the stock at the closing of the trading. Both values can be same, or not.

Data set is daily hisorical prices for 10 years (Jul 24, 2007 - Jul 24, 2017),
which is 2518 dataset for each stock (2518 days of trading).

80% of the data set were used for training.<br />
20% of the data set were used for testing.

Download the CSV file for each (GE, S&P 500, Microsoft, Apple, Toyota)

- GE: https://finance.yahoo.com/quote/GE/history?p=GE
- Microsoft: https://finance.yahoo.com/quote/MSFT/history?p=MSFT
- Apple: https://finance.yahoo.com/quote/AAPL/history?p=AAPL
- Toyota: https://finance.yahoo.com/quote/TM/history?p=TM
- S&P 500: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC


**_Example data from csv_**

The data includes following properties:

| Date       | Open       | High       | Low	    | Close      | Adj Close | Volume  |
| ---------- | ---------- | ---------- | ---------- | ---------- | --------- | ------- |
| 2007-07-24 | 123.779999 | 123.779999 | 122.309998	| 122.489998 | 98.390572 | 418000  |
| 2007-07-25 | 123.389999 | 123.410004 | 121.500000	| 122.290001 | 98.229897 | 557900  |
| 2007-07-26 | 122.320000 | 122.349998 | 117.050003	| 119.199997 | 95.747864 | 1258500 |

The data set is straight forward and there is no missing value in each column.


### Exploratory Visualization

**Toyota Motor data**

Plot of the dataset from Toyota Motor for Adjusted Close Price for 10 years trading period.

![alt text](images/tm_data_viz.png "Toyota Motor")


### Algorithms and Techniques

From the data set of yahoo finance. Predict the closing price of a target day based on the 
historical data up to the previous day of the target day.

I try to use a kind of Recurrent Neural Network (RNN), called Long Short Term Memory (LSTM) from Keras library for the solution model. RNN is a deep learning algorithm that has a "memory"
to remember / store the information about what has been calculated previously.

LSTM networks have memory blocks that are connected through layers, and it can choose what it remembers and can decide to forget, so it can adjust how much of memory it should pass to next layer.

Since I use the time series of data of stock prices and try to predict the price,
LSTM looks good fits for this project.

### Benchmark

As a baseline benchmark model, I used **Linear Regression** model.

**_Why I chose Linear Regression as a baseline benchmark model?_**

Linear Regression is simple and fairly rigid approximeter to be used as a baseline algorithm.
Since I do not want to set the baseline model to be complicated, slow, or requiring a sort of data transformation
to implement. Linear Regression is simple, fast, and is not required to transform the dataset. So,
it satisfies my need for this.

As the solution model, I chose **LSTM** model as the solution benchmark model.

## III. Methodology

### Data Preprocessing

Since it simply tries to predicts the **Adjusted Close** Price from the past data, I believe
there is no need for feature engineering.

So I took the **Adj Close** column from the dataset.

For Linear Regression,
I took the Adjusted Closing Price to put them in linear regression line.


For LSTM model,
I normalized the Adjusted Closing Prices to improve the convergence.
I used LSTM from Keras libaray with Tensorflow backend.


TODO: Explain helper functions, why normalized for LSTM?

### Implementation

**Baseline - Linear Regression model**

TODO: Explain the code

**The solution - LSTM model**

TODO: Explain the code

**The accuracy**

**Toyota**<br />
Linear Regression MSE: <br />
LSTM MSE: <br />
Linear Regression RMSE: <br />
LSTM RMSE: <br />

**Apple**<br />
Linear Regression MSE: <br />
LSTM MSE: <br />
Linear Regression RMSE: <br />
LSTM RMSE: <br />

**GE**<br />
Linear Regression MSE: <br />
LSTM MSE: <br />
Linear Regression RMSE: <br />
LSTM RMSE: <br />

**Microsoft**<br />
Linear Regression MSE: <br />
LSTM MSE: <br />
Linear Regression RMSE: <br />
LSTM RMSE: <br />

**S&P 500**<br />
Linear Regression MSE: <br />
LSTM MSE: <br />
Linear Regression RMSE: <br />
LSTM RMSE: <br />


### Refinement (TODO)

TODO: Explain the code that changed (parameter tunings, etc)


## IV. Results (TODO)

### Model Evaluation and Validation (TODO)

After improvement, the final model predicts with higher accuracy than basic model.

TODO: place images of plots and put the final results, how it is valid? how justified for unseen data? params appropriate?


### Justification (TODO)

Compared to the simple linear regression model (benchmark model), LSTM model (solution model) predicts better.

TODO: comparison, significant enough to solve the original issue?


## V. Conclusion (TODO)

### Free-Form Visualization (TODO)

Add images here!


### Reflection (TODO)



In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement (TODO)



In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

### References

- http://fortune.com/2017/03/30/blackrock-robots-layoffs-artificial-intelligence-ai-hedge-fund/
- https://seekingalpha.com/article/4083754-superior-portfolio-roi-artificially-intelligent-algorithms
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
- http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
- http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
- https://www.freelancermap.com/freelancer-tips/11865-trend-prediction-with-lstm-rnns-using-keras-tensorflow-in-3-steps]
- https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02
- http://www.naun.org/main/NAUN/mcs/2017/a042002-041.pdf
- https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning
- Machine Learning for Trading: https://www.udacity.com/course/machine-learning-for-trading--ud501
- Time Series Forecasting https://www.udacity.com/course/time-series-forecasting--ud980