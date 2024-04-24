Intro to Time Series
================
Riley M. Anderson
April 24, 2024

  

- [Time Series Analysis](#time-series-analysis)
  - [Why use time-series?](#why-use-time-series)
  - [Autocorrelation & Autocovariance](#autocorrelation--autocovariance)
- [Stationarity vs Nonstationarity](#stationarity-vs-nonstationarity)
  - [Transforming for stationarity](#transforming-for-stationarity)
    - [Differencing](#differencing)
    - [Detrending](#detrending)
- [Basic Model Types: AR(p), MA(q), ARMA(p,q), ARIMA(p,d,q),
  Decomposition](#basic-model-types-arp-maq-armapq-arimapdq-decomposition)
  - [Autoregressive AR(p) Models](#autoregressive-arp-models)
  - [Moving Average MA(q) Models](#moving-average-maq-models)
  - [Autoregressive Moving Average ARMA(p,q)
    Models](#autoregressive-moving-average-armapq-models)
  - [Autoregressive Integrated Moving Average ARIMA(p,d,q)
    Models](#autoregressive-integrated-moving-average-arimapdq-models)
  - [Decomposition Models](#decomposition-models)
- [Fitting AR/MA/ARMA/ARIMA models with the Box Jenkins
  Method](#fitting-armaarmaarima-models-with-the-box-jenkins-method)
  - [Fit some time series with real
    data](#fit-some-time-series-with-real-data)
    - [Check for stationarity](#check-for-stationarity)
  - [Transforming for Stationarity & Identifying Model
    Parameters](#transforming-for-stationarity--identifying-model-parameters)
  - [Checking the Residuals of the Model Fit (Ljung-Box
    test)](#checking-the-residuals-of-the-model-fit-ljung-box-test)
  - [Making a forecast for each
    model](#making-a-forecast-for-each-model)
- [Fitting Seasonal Trend Loess (STL) Decomposition
  Models](#fitting-seasonal-trend-loess-stl-decomposition-models)
- [Exponential Smoothing](#exponential-smoothing)
- [Simple Exponential Smoothing](#simple-exponential-smoothing)
  - [Validation of SES](#validation-of-ses)
- [Holt’s method](#holts-method)
- [Holt-Winter’s Seasonal method](#holt-winters-seasonal-method)
  - [Optimize the Holt-Winter’s additive
    model](#optimize-the-holt-winters-additive-model)
- [Damping model](#damping-model)
- [Where to go Next](#where-to-go-next)
- [Session Information](#session-information)

# Time Series Analysis

(Univariate) time series data is defined as sequence data over time:
![X\_{1}](https://latex.codecogs.com/png.latex?X_%7B1%7D "X_{1}"),
![X\_{2}](https://latex.codecogs.com/png.latex?X_%7B2%7D "X_{2}"),…,
![X\_{T}](https://latex.codecogs.com/png.latex?X_%7BT%7D "X_{T}")

where *T* is the time period and
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") is the
value of the time series at a particular point.

Examples: daily temperatures in Boston, US presidential election turnout
by year, minute stock prices

Variables in time series models generally fall into three categories:

1)  endogenous (past values of the time-series in questions)

2)  random noise (accounting for uncertainty (CIs), and past random
    shocks)

3)  exogenous (predictors outside the time-series)

All time series models involve (1) and (2) but (3) is optional.

## Why use time-series?

1)  many forecasting tasks actually involve small samples which makes
    machine learning less effective

2)  time series models are more interpretable and less black box than
    machine learning algorithms

3)  time series appropriately accounts for forecasting uncertainty.

As an example, lets look at the following data generating process known
as a random walk:
![X\_{t} = X\_{t-1} + \varepsilon\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20X_%7Bt-1%7D%20%2B%20%5Cvarepsilon_%7Bt%7D "X_{t} = X_{t-1} + \varepsilon_{t}")

We can compare the forecasting performance of linear regression to that
of a basic time series model known as an AR(1) model.

![](IntroToTimeSeries_files/figure-gfm/compare_models-1.png)<!-- -->

Linear regression overfits the past observations and doesn’t account for
compounding error. With successive future predictions, the confidence
interval should increase over time. Too much weight on random error.

AR model just forecasts the average
![X](https://latex.codecogs.com/png.latex?X "X") values from the past
observations.

## Autocorrelation & Autocovariance

Autocorrelation/autocovariance refers to the correlation/covariance
between two observations in the time series at different points.

The central idea behind it is how related the data/time series is over
time.

For ease of interpretation we typically focus on autocorrelation
i.e. what is the correlation between
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") and
![X\_{t + p}](https://latex.codecogs.com/png.latex?X_%7Bt%20%2B%20p%7D "X_{t + p}")
for some integer *p*. (Correlation between all *p* periods about in the
sample)

A related concept is partial autocorrelation that computes the
correlation adjusting for previous lags/periods i.e. the autocorrelation
between
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") and
![X\_{t + p}](https://latex.codecogs.com/png.latex?X_%7Bt%20%2B%20p%7D "X_{t + p}")
adjusting for the correlation of
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") and
![X\_{t + 1}](https://latex.codecogs.com/png.latex?X_%7Bt%20%2B%201%7D "X_{t + 1}"),
… ,
![X\_{t + p - 1}](https://latex.codecogs.com/png.latex?X_%7Bt%20%2B%20p%20-%201%7D "X_{t + p - 1}").

When analyzing time series we usually view autocorrelation/partial
autocorrelation in ACF/PACF plots.

Let’s view this for the random walk model we analyzed above:
![X\_{t} = X\_{t-1} + \varepsilon\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20X_%7Bt-1%7D%20%2B%20%5Cvarepsilon_%7Bt%7D "X_{t} = X_{t-1} + \varepsilon_{t}")

``` r
dat <- sim.random.walk(n = 100)

#plot random walk
sim.plot <- dat %>% ggplot(aes(t,X)) +
  geom_line() +
  xlab("T") +
  ylab("X") +
  ggtitle("Time Series Plot")

#ACF plot
sim.corr <- ggAcf(dat$X,type="correlation") +
  ggtitle("Autocorrelation ACF Plot")

#PACF plot
sim.partial <- ggAcf(dat$X,type="partial") +
  ggtitle("Partial Autocorrelation PACF Plot")

grid.arrange(sim.plot, sim.corr, sim.partial)
```

![](IntroToTimeSeries_files/figure-gfm/simulate_random_walk-1.png)<!-- -->

The ACF plot shows a high degree of correlation between each successive
lag. This makes sense because we simulated this data with a random walk,
where each successive time point value
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") is a
function of the previous time point value
![X\_{t-1}](https://latex.codecogs.com/png.latex?X_%7Bt-1%7D "X_{t-1}").

In contrast, the PACF plot shows the correlation between lags
![p\_{1},...,n](https://latex.codecogs.com/png.latex?p_%7B1%7D%2C...%2Cn "p_{1},...,n").
The correlation is high when the lag is 1, but is weak for lags \> 1
(i.e., lags with greater distance in time points).

# Stationarity vs Nonstationarity

Is the distribution of the data over time consistent?

There are two main forms of stationarity.

- Strict stationarity imples:

  - The cumulative distribution function of the data does not depend on
    time:

![F\_{X}(X\_{1},...,X\_{T}) = F\_{X}(X\_{1 + \Delta},...,X\_{T + \Delta}) \forall \Delta \in \mathbb{R}](https://latex.codecogs.com/png.latex?F_%7BX%7D%28X_%7B1%7D%2C...%2CX_%7BT%7D%29%20%3D%20F_%7BX%7D%28X_%7B1%20%2B%20%5CDelta%7D%2C...%2CX_%7BT%20%2B%20%5CDelta%7D%29%20%5Cforall%20%5CDelta%20%5Cin%20%5Cmathbb%7BR%7D "F_{X}(X_{1},...,X_{T}) = F_{X}(X_{1 + \Delta},...,X_{T + \Delta}) \forall \Delta \in \mathbb{R}")

- Weak stationarity implies:

  - the mean of the time series is constant
    ![E(X\_{T}) = E(X\_{t + \Delta})](https://latex.codecogs.com/png.latex?E%28X_%7BT%7D%29%20%3D%20E%28X_%7Bt%20%2B%20%5CDelta%7D%29 "E(X_{T}) = E(X_{t + \Delta})")

  - the autocovariance/autocorrelation only depends on the time
    difference between points
    ![ACF(X\_{t}, X\_{t + \Delta - 1}) = ACF(X\_{1}, X\_{\Delta})](https://latex.codecogs.com/png.latex?ACF%28X_%7Bt%7D%2C%20X_%7Bt%20%2B%20%5CDelta%20-%201%7D%29%20%3D%20ACF%28X_%7B1%7D%2C%20X_%7B%5CDelta%7D%29 "ACF(X_{t}, X_{t + \Delta - 1}) = ACF(X_{1}, X_{\Delta})")

  - the time series has a finite variance
    ![Var(X\_{\Delta}) \< \infty \forall \Delta \in \mathbb{R}](https://latex.codecogs.com/png.latex?Var%28X_%7B%5CDelta%7D%29%20%3C%20%5Cinfty%20%5Cforall%20%5CDelta%20%5Cin%20%5Cmathbb%7BR%7D "Var(X_{\Delta}) < \infty \forall \Delta \in \mathbb{R}")

``` r

df<-sim.stationary.example(n=1000)

head(df)
##   t          X1        X2          X3
## 1 1 -0.44377902 0.2318932  1.28923628
## 2 2  2.24020140 2.4106363 -1.63588767
## 3 3  1.93954222 3.6174854 -0.08243868
## 4 4  1.68071924 5.9578636 -1.08716585
## 5 5  0.42455859 4.3604708  0.97287262
## 6 6 -0.03806078 6.6450558 -1.87240895

gns <- ggplot(df, aes(x = t, y = X1)) +
  geom_line() +
  labs(x = "T", y = "X1", title = "Nonstationary")

gs <- ggplot(df, aes(x = t, y = X3)) +
  geom_line() +
  labs(x = "T", y = "X3", title = "Stationary")

grid.arrange(gns, gs)
```

![](IntroToTimeSeries_files/figure-gfm/station_v_nonstation-1.png)<!-- -->

The nonstationary time series has an unstable mean, that changes over
time. Whereas the stationary time series appears to have a constant mean
over time, and finite variance.

``` r

#ACF for nonstationary and stationary time series
nsACF <- ggAcf(df$X1, type = "correlation") +
  labs(x = "T", y = "X1", title = "Nonstationary")

sACF <- ggAcf(df$X3, type = "correlation") +
  labs(x = "T", y = "X3", title = "Stationary")

grid.arrange(nsACF, sACF)
```

![](IntroToTimeSeries_files/figure-gfm/ACF_plots_S_vs_NS-1.png)<!-- -->

The nonstationary time series has all significant correlations across
lags and shows non mean-reverting behavior.

The stationary time series has non-significant correlations between
almost all lags.

We can perform a unit root test (Augmented Dickey-Fuller test) to test
for stationarity:

``` r

adf.test(df$X1)
## 
##  Augmented Dickey-Fuller Test
## 
## data:  df$X1
## Dickey-Fuller = -1.4568, Lag order = 9, p-value = 0.8083
## alternative hypothesis: stationary
```

In the ADF test, the null hypothesis is *nonstationary*, while the
alternative hypothesis is *stationary*. For the above time series, we
fail to reject to null hypothesis (*P* = 0.808) and conclude that the
time series exhibits nonstationarity.

``` r

adf.test(df$X3)
## 
##  Augmented Dickey-Fuller Test
## 
## data:  df$X3
## Dickey-Fuller = -9.0269, Lag order = 9, p-value = 0.01
## alternative hypothesis: stationary
```

In the stationary example, the ADF test confirms stationarity by
rejecting the null hypothesis (*P* = 0.01).

## Transforming for stationarity

Typically, nonstationary data are transformed by differencing or
detrending to make the data more stationary.

### Differencing

Differencing involves taking differences between successive time series
values.

The order of differencing is defined as *p* for
![X\_{t} - X\_{t-p}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20-%20X_%7Bt-p%7D "X_{t} - X_{t-p}").

Let’s transform a nonstationary time series to stationary by
differencing with the random walk model.

In a random walk
![X\_{t} = X\_{t-1} + \varepsilon\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20X_%7Bt-1%7D%20%2B%20%5Cvarepsilon_%7Bt%7D "X_{t} = X_{t-1} + \varepsilon_{t}")
where
![\varepsilon\_{t} \sim N(0,\sigma^{2}) iid](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D%20%5Csim%20N%280%2C%5Csigma%5E%7B2%7D%29%20iid "\varepsilon_{t} \sim N(0,\sigma^{2}) iid").

Differencing with an order of one means that
![\tilde{X}\_{t} = X\_{t}-X\_{t-1} = \varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Ctilde%7BX%7D_%7Bt%7D%20%3D%20X_%7Bt%7D-X_%7Bt-1%7D%20%3D%20%5Cvarepsilon_%7Bt%7D "\tilde{X}_{t} = X_{t}-X_{t-1} = \varepsilon_{t}").

With data generated by random walk, the differencing order is t-1, which
if we difference by this order, we have removed the
![X\_{t}-X\_{t-1}](https://latex.codecogs.com/png.latex?X_%7Bt%7D-X_%7Bt-1%7D "X_{t}-X_{t-1}")
term and are left with the random noise term
![\varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D "\varepsilon_{t}"),
which is by definition, stationary.

``` r

# differencing with an order of 1
diff <- df$X1 - lag(df$X1, 1) # for all values of X1, subtract the previous (1th) value


#plot original and differenced time series
untrans <- ggAcf(df$X1, type = "correlation")

lag.trans <- ggAcf(diff, type = "correlation")

grid.arrange(untrans, lag.trans)
```

![](IntroToTimeSeries_files/figure-gfm/differencing_in_time_series-1.png)<!-- -->

The ACF for df\$X1 (generated by random walk), does not die off quickly,
suggesting high correlations between each successive term (lag).

However, the lag transformed (1 lag difference between time points) has
weak correlations between these lags, showing stationarity.

### Detrending

An alternative way to induce stationarity from nonstationary data.

Detrending involves removing a deterministic relationship with time.

As an example suppose we have the following data generating process
![X\_{t} = B\_{t} + \varepsilon\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20B_%7Bt%7D%20%2B%20%5Cvarepsilon_%7Bt%7D "X_{t} = B_{t} + \varepsilon_{t}")
where
![\varepsilon\_{t} = N(0,\sigma^{2})iid](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D%20%3D%20N%280%2C%5Csigma%5E%7B2%7D%29iid "\varepsilon_{t} = N(0,\sigma^{2})iid").

Here, we have a deterministic linear relationship with time, and
removing that relationship would leave the transformed time series
![\tilde{X}\_{t}](https://latex.codecogs.com/png.latex?%5Ctilde%7BX%7D_%7Bt%7D "\tilde{X}_{t}")
where the only predictor is the random noise term
![\varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D "\varepsilon_{t}"),
which is by definition stationary with constant mean and finite
variance.

Detrending involves using the transformed time series
![\tilde{X}\_{t} = X\_{t}-B\_{t} = \varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Ctilde%7BX%7D_%7Bt%7D%20%3D%20X_%7Bt%7D-B_%7Bt%7D%20%3D%20%5Cvarepsilon_%7Bt%7D "\tilde{X}_{t} = X_{t}-B_{t} = \varepsilon_{t}").

In the real world, we don’t know the actual relationship with time, so
we build a model and then remove the time component
(![B\_{t}](https://latex.codecogs.com/png.latex?B_%7Bt%7D "B_{t}"))
leaving only the random noise term (residual error
![\varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D "\varepsilon_{t}")).

``` r

detrended <- resid(lm(X2 ~ t, data = df))

#plot original and detrended time series
untransX2 <- ggAcf(df$X2, type = "correlation")

detrendedX2 <- ggAcf(detrended, type = "correlation")

grid.arrange(untransX2, detrendedX2)
```

![](IntroToTimeSeries_files/figure-gfm/detrending_with_linear_regression-1.png)<!-- -->

The original data exhibit nonstationary behavior, with correlations
extending far beyond the confidence intervals (highly significant).

Alternatively, the detrended time series correlations fall mainly within
the confidence intervals (not significant) and exhibit stationary
behavior.

# Basic Model Types: AR(p), MA(q), ARMA(p,q), ARIMA(p,d,q), Decomposition

## Autoregressive AR(p) Models

AR models specify
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") as a
function of lagged time series values
![X\_{t-1}, X\_{t-2},...](https://latex.codecogs.com/png.latex?X_%7Bt-1%7D%2C%20X_%7Bt-2%7D%2C... "X_{t-1}, X_{t-2},...")
i.e
![X\_{t} = \mu + \phi\_{1}X\_{t-p}+...+\phi\_{p}X\_{t-p}+\varepsilon\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20%5Cmu%20%2B%20%5Cphi_%7B1%7DX_%7Bt-p%7D%2B...%2B%5Cphi_%7Bp%7DX_%7Bt-p%7D%2B%5Cvarepsilon_%7Bt%7D "X_{t} = \mu + \phi_{1}X_{t-p}+...+\phi_{p}X_{t-p}+\varepsilon_{t}")

where ![\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") is a mean
term and
![\varepsilon\_{t} \sim N(0,\sigma^{2})](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D%20%5Csim%20N%280%2C%5Csigma%5E%7B2%7D%29 "\varepsilon_{t} \sim N(0,\sigma^{2})")
is a random error.
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") is a a
function of the ![\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu")
(mean/intercept) term plus a linear combination of past lagged values.
Our model will estimate the
![\phi\_{i}](https://latex.codecogs.com/png.latex?%5Cphi_%7Bi%7D "\phi_{i}")
parameters using one of several methods (Yule-Walker equations, Maximum
Likelihood, or even linear regression).

AR models can sometimes be non-stationary if a unit root is present
(rejected the null in an ADF test).

In AR models, because we depend on past lagged
![X\_{i}](https://latex.codecogs.com/png.latex?X_%7Bi%7D "X_{i}")
values, random noise terms in past periods can have influence on the
future predictions. Conceptually, this is why the process can be
nonstationary. Past large shocks can cause a series to diverge and
become non mean-reverting and have infinite variance.

When fitting an AR model the key choice is *p*, the number of lags (and
![\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi") parameters)
to include in the model.

## Moving Average MA(q) Models

Similar to AR models, but instead of using past lagged values of the
time series, we use past lagged random shock events (random error terms)

MA models specify
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") using
random noise lags:

![X\_{t} = \mu + \varepsilon\_{t} + \Theta\_{1}\varepsilon\_{t-1}+...+\Theta\_{q}\varepsilon\_{t-q}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20%5Cmu%20%2B%20%5Cvarepsilon_%7Bt%7D%20%2B%20%5CTheta_%7B1%7D%5Cvarepsilon_%7Bt-1%7D%2B...%2B%5CTheta_%7Bq%7D%5Cvarepsilon_%7Bt-q%7D "X_{t} = \mu + \varepsilon_{t} + \Theta_{1}\varepsilon_{t-1}+...+\Theta_{q}\varepsilon_{t-q}")

where ![\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") is a mean
term and
![\varepsilon\_{t} \sim N(0,\sigma^{2})](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D%20%5Csim%20N%280%2C%5Csigma%5E%7B2%7D%29 "\varepsilon_{t} \sim N(0,\sigma^{2})")
is a random error.
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") is a a
function of the ![\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu")
(mean/intercept) term plus the random noise term in the current period
![\varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D "\varepsilon_{t}")
plus a linear combination of the past random noise terms
![\Theta\_{1,...,p}](https://latex.codecogs.com/png.latex?%5CTheta_%7B1%2C...%2Cp%7D "\Theta_{1,...,p}").
Our model will estimate the
![\phi\_{i}](https://latex.codecogs.com/png.latex?%5Cphi_%7Bi%7D "\phi_{i}")
parameters using one of several methods

Because MA models specify
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") as a
function of past random noise terms, we don’t actually observe these
past random noise terms, fitting them is more complicated and involves
interative fitting. Also, because MA models are a linear combination of
past random shocks, they are by construct, stationary (random noise
terms are independent and identically distributed, *iid*, constant mean
and finite variance).

![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") is a
function of a finite number of random lags, past random shocks can only
have a finite influence on future periods. This is why MA models are
always stationary, whatever influence the random shocks have will trend
toward zero over time.

Similar to an AR model, when fitting an MA model the key choice is *q*,
the number of random shock lags.

## Autoregressive Moving Average ARMA(p,q) Models

ARMA(p,q) models are a combination of an AR and MA model:

![X\_{t} = \mu + \phi\_{1}X\_{t-1}+...+\phi\_{p}X\_{t-p} + \varepsilon\_{t} + \Theta\_{1}\varepsilon\_{t-1} +...+\Theta\_{q}\varepsilon\_{t-q}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20%5Cmu%20%2B%20%5Cphi_%7B1%7DX_%7Bt-1%7D%2B...%2B%5Cphi_%7Bp%7DX_%7Bt-p%7D%20%2B%20%5Cvarepsilon_%7Bt%7D%20%2B%20%5CTheta_%7B1%7D%5Cvarepsilon_%7Bt-1%7D%20%2B...%2B%5CTheta_%7Bq%7D%5Cvarepsilon_%7Bt-q%7D "X_{t} = \mu + \phi_{1}X_{t-1}+...+\phi_{p}X_{t-p} + \varepsilon_{t} + \Theta_{1}\varepsilon_{t-1} +...+\Theta_{q}\varepsilon_{t-q}")

where ![\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu") is a mean
term and
![\varepsilon\_{t} \sim N(0,\sigma^{2})](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D%20%5Csim%20N%280%2C%5Csigma%5E%7B2%7D%29 "\varepsilon_{t} \sim N(0,\sigma^{2})")
is a random error.
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") is a
function of a ![\mu](https://latex.codecogs.com/png.latex?%5Cmu "\mu")
(mean/intercept) term plus a random noise term in the current period
![\varepsilon\_{t}](https://latex.codecogs.com/png.latex?%5Cvarepsilon_%7Bt%7D "\varepsilon_{t}")
plus a linear combination of both past lagged time series values and
also past random noise terms

When fitting an ARMA model, we need to choose two things: p, the number
of AR lags, and q, the number of MA lags.

## Autoregressive Integrated Moving Average ARIMA(p,d,q) Models

ARIMA(p,d,q) is an ARMA model with differencing.

Take an ARMA model and add in differencing. We take the difference
between the time series at successive points in time.

When fitting an ARIMA model we need to choose three things: p, the
number of AR lags, q, the number of MA lags, and d, the number of
differences to use.

## Decomposition Models

Decomposition models specify
![X\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D "X_{t}") as a
combination of a trend component
(![T\_{t}](https://latex.codecogs.com/png.latex?T_%7Bt%7D "T_{t}")),
seasonal component
(![S\_{t}](https://latex.codecogs.com/png.latex?S_%7Bt%7D "S_{t}")), and
an error component/residual
(![E\_{t}](https://latex.codecogs.com/png.latex?E_%7Bt%7D "E_{t}"))
i.e. ![X\_{t} = f(T\_{t},S\_{t},E\_{t})](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20f%28T_%7Bt%7D%2CS_%7Bt%7D%2CE_%7Bt%7D%29 "X_{t} = f(T_{t},S_{t},E_{t})").

Common decomposition forms are:
![X\_{t} = T\_{t}+S\_{t}+E\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20T_%7Bt%7D%2BS_%7Bt%7D%2BE_%7Bt%7D "X_{t} = T_{t}+S_{t}+E_{t}")
or
![X\_{t} = T\_{t}\*S\_{t}\*E\_{t}](https://latex.codecogs.com/png.latex?X_%7Bt%7D%20%3D%20T_%7Bt%7D%2AS_%7Bt%7D%2AE_%7Bt%7D "X_{t} = T_{t}*S_{t}*E_{t}")
(where we then take logs to recover the additive form).

There are various ways to estimate the different trend components:
exponential smoothing, state space models/Kalman filtering, STL models,
etc.

# Fitting AR/MA/ARMA/ARIMA models with the Box Jenkins Method

How to fit AR/MA/ARMA/ARIMA models on a real data set and review a
generic strategy for fitting them (Box Jenkins method).

This process involves several steps to help identify the *p*, *d*, and
*q* parameters that we need:

- Identify whether the time series is stationary or not

- Identify *p*, *d*, and *q* of the time series by

  - Making the time series stationary through differencing/detrending to
    find *d*
  - Looking at ACF/PACF to find *p* and *q*
  - Using model fit diagnostics like AIC or BIC to select the best model
    to find *p*, *d*, and *q*

- Check the model fit using the Ljung-Box test

## Fit some time series with real data

This data comes from the FRED database (St. Louis Federal Reserve) and
describes the monthly unemployment rate for Massachusetts between
January 1976 and January 2020.

            DATE MAURN
    1 1976-01-01  11.6
    2 1976-02-01  11.3
    3 1976-03-01  10.9
    4 1976-04-01   9.9
    5 1976-05-01   9.4
    6 1976-06-01   9.8

Where MAURN is the monthly unemployment rate.

### Check for stationarity

``` r

ur %>% 
  ggplot(aes(x = DATE, y = MAURN)) +
  geom_line()
```

![](IntroToTimeSeries_files/figure-gfm/ur_stationarity_plot-1.png)<!-- -->

Doesn’t look stationary, mean varies over time.

``` r

ggAcf(ur$MAURN, type = "correlation")
```

![](IntroToTimeSeries_files/figure-gfm/ur_stationarity_ACF-1.png)<!-- -->
Again, looks nonstationary.

``` r

adf.test(ur$MAURN)
## 
##  Augmented Dickey-Fuller Test
## 
## data:  ur$MAURN
## Dickey-Fuller = -3.0954, Lag order = 8, p-value = 0.1146
## alternative hypothesis: stationary
```

Fail to reject the null hypothesis of non-stationarity.

## Transforming for Stationarity & Identifying Model Parameters

``` r

ar.mod <- auto.arima(ur$MAURN,
                     max.d = 0, # no differencing
                     max.q = 0, # no random noise terms
                     allowdrift = T #include a mu term
                     # note only p is allowed to vary -> AR model
                     # p is the number of autoregressive lags
                     )

ar.mod
## Series: ur$MAURN 
## ARIMA(1,0,0) with non-zero mean 
## 
## Coefficients:
##          ar1    mean
##       0.9787  5.7425
## s.e.  0.0101  0.8498
## 
## sigma^2 = 0.2:  log likelihood = -325.44
## AIC=656.88   AICc=656.93   BIC=669.7
```

``` r

ma.mod <- auto.arima(ur$MAURN,
                     max.d = 0, # no differencing
                     max.p = 0, # no lags
                     allowdrift = T #include a mu term
                     # note only q is allowed to vary -> MA model
                     # q is the number of random noise (shock lag) events
                     )

ma.mod
## Series: ur$MAURN 
## ARIMA(0,0,5) with non-zero mean 
## 
## Coefficients:
##          ma1     ma2     ma3     ma4     ma5    mean
##       1.3646  1.7103  1.4882  1.2714  0.4804  5.4588
## s.e.  0.0368  0.0492  0.0578  0.0393  0.0350  0.1507
## 
## sigma^2 = 0.229:  log likelihood = -361.03
## AIC=736.05   AICc=736.27   BIC=765.95
```

``` r

arma.mod <- auto.arima(ur$MAURN,
                     max.d = 0, # no differencing
                     allowdrift = T #include a mu term
                     # note p and q are allowed to vary -> ARMA model
                     # p is the number of autoregressive lags
                     # q is the number of random noise (shock lag) events
                     )

arma.mod
## Series: ur$MAURN 
## ARIMA(3,0,2) with non-zero mean 
## 
## Coefficients:
##           ar1     ar2     ar3     ma1     ma2    mean
##       -0.2267  0.5998  0.5573  1.3361  0.8876  5.7038
## s.e.   0.0885  0.0544  0.0569  0.0544  0.0221  0.7764
## 
## sigma^2 = 0.1693:  log likelihood = -280.15
## AIC=574.3   AICc=574.51   BIC=604.19
```

``` r

arima.mod <- auto.arima(ur$MAURN,
                     allowdrift = T #include a mu term
                     # note d, p, and q are allowed to vary -> ARIMA model
                     # d is the 
                     # p is the number of autoregressive lags
                     # q is the number of random noise (shock lag) events
                     )

arima.mod
## Series: ur$MAURN 
## ARIMA(4,1,2) 
## 
## Coefficients:
##          ar1      ar2      ar3     ar4      ma1     ma2
##       1.0029  -0.1834  -0.3982  0.4872  -1.1149  0.2512
## s.e.  0.0708   0.0750   0.0560  0.0394   0.0793  0.0711
## 
## sigma^2 = 0.1509:  log likelihood = -247.45
## AIC=508.9   AICc=509.12   BIC=538.78
```

## Checking the Residuals of the Model Fit (Ljung-Box test)

``` r

#calculate residuals of each model
ar.resid <- resid(ar.mod)
ma.resid <- resid(ma.mod)
arma.resid <- resid(arma.mod)
arima.resid <- resid(arima.mod)
```

``` r

#plot PACF plot of each models residuals
ggAcf(ar.resid, type = "partial")
```

![](IntroToTimeSeries_files/figure-gfm/ar_resid_plot-1.png)<!-- -->

``` r

Box.test(ar.resid, type = "Ljung-Box", lag = 1)
## 
##  Box-Ljung test
## 
## data:  ar.resid
## X-squared = 0.88802, df = 1, p-value = 0.346
```

``` r

#plot PACF plot of each models residuals
ggAcf(ma.resid, type = "partial")
```

![](IntroToTimeSeries_files/figure-gfm/ma_resid_plot-1.png)<!-- -->

``` r

Box.test(ma.resid, type = "Ljung-Box", lag = 1)
## 
##  Box-Ljung test
## 
## data:  ma.resid
## X-squared = 0.56386, df = 1, p-value = 0.4527
```

``` r

#plot PACF plot of each models residuals
ggAcf(arma.resid, type = "partial")
```

![](IntroToTimeSeries_files/figure-gfm/arma_resid_plot-1.png)<!-- -->

``` r

Box.test(arma.resid, type = "Ljung-Box", lag = 1)
## 
##  Box-Ljung test
## 
## data:  arma.resid
## X-squared = 0.96747, df = 1, p-value = 0.3253
```

``` r

#plot PACF plot of each models residuals
ggAcf(arima.resid, type = "partial")
```

![](IntroToTimeSeries_files/figure-gfm/arima_resid_plot-1.png)<!-- -->

``` r

Box.test(arima.resid, type = "Ljung-Box", lag = 1)
## 
##  Box-Ljung test
## 
## data:  arima.resid
## X-squared = 0.0032696, df = 1, p-value = 0.9544
```

## Making a forecast for each model

``` r

#make forecast for each model
ar.fc <- forecast(ar.mod, h = 24, level = 80)
ma.fc <- forecast(ma.mod, h = 24, level = 80)
arma.fc <- forecast(arma.mod, h = 24, level = 80)
arima.fc <- forecast(arima.mod, h = 24, level = 80)


#plot forecast for each model
g1 <- autoplot(ar.fc)
g2 <- autoplot(ma.fc)
g3 <- autoplot(arma.fc)
g4 <- autoplot(arima.fc)
grid.arrange(g1, g2, g3, g4)
```

![](IntroToTimeSeries_files/figure-gfm/forecasts_arima-1.png)<!-- -->

# Fitting Seasonal Trend Loess (STL) Decomposition Models

``` r

#transform to time series object; need to specify frequency
ur.ts <- ts(ur$MAURN, frequency = 12) # monthly data (12 months/year)

#fit stil model
stl.mod <- stl(ur.ts, s.window = "periodic")


#plot model fit
autoplot(stl.mod)
```

![](IntroToTimeSeries_files/figure-gfm/STL_model-1.png)<!-- -->

``` r
#make forecast
stl.fc <- forecast(stl.mod, h = 24, level = 80)

autoplot(stl.fc)
```

![](IntroToTimeSeries_files/figure-gfm/STL_forecast-1.png)<!-- -->

# Exponential Smoothing

- From
  <https://www.geeksforgeeks.org/exponential-smoothing-in-r-programming/#>

Exponential smoothing is the oldest version of time series analysis and
is most useful when we want more recent events to have a greater
influence on our forecasts than more distant events (i.e., inverse time
weighting).

This model works by assigning exponentially decreasing weights to events
further in the past. Additionally, the “smoothing” component acts as a
low-pass filter, generally smoothing out the trend over time by removing
high-frequency noise.

Basic exponential smoothing models include:

- Simple Exponential Smoothing
- Holt’s method
- Holt-Winter’s Seasonal method
- Damped Trend method

# Simple Exponential Smoothing

- Used for data that has no trend or seasonal pattern

Weight of every parameter is determined by a smoothing parameter,
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") with
range 0-1, but in practice is usually between 0.1 and 0.2. When
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") is
closer to 0, historical events will be weighted similar to recent events
(“slow-learning”).
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") close
to 1 means extreme weighting on recent event where historical values
have little influence on future predictions (“fast-learning”).

An example using Google’s stock price data

``` r

# training data:
google.train <- window(goog, end = 900)

# test data:
google.test <- window(goog, start = 901)

# Performing SES on  the 
# Google stock data
ses.goog <- ses(google.train, 
                alpha = .2,
                h = 100) # h is number of time points into the future
autoplot(ses.goog)
```

![](IntroToTimeSeries_files/figure-gfm/SES_google-1.png)<!-- -->

The flat line forecast suggests that the model is not capturing the
present trend.

We can remove the trend with the diff() function.

``` r

google.diff <- diff(google.train)

#autoplot(google.diff)

ses.goog.diff <- ses(google.diff,
                     alpha = 0.2,
                     h = 100)

autoplot(ses.goog.diff)
```

![](IntroToTimeSeries_files/figure-gfm/SES_google_diff-1.png)<!-- -->

## Validation of SES

Create a differenced validation data set to compare with the differenced
training set. Then we’ll iterate the model fit through all values
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") 0.01 -
0.99 to find the optimal value that minimizes RMSE.

``` r

# remove the trend from the test data
goog.diff.test <- diff(google.test)

accuracy(ses.goog.diff, goog.diff.test)[2,2]
## [1] 8.14145

# comparing our model
alpha <- seq(.01, .99, by = .01)
RMSE <- NA
for(i in seq_along(alpha)) {
  fit <- ses(google.diff, alpha = alpha[i],
             h = 100)
  RMSE[i] <- accuracy(fit, 
                      goog.diff.test)[2,2]
}

# convert to a data frame and 
# identify min alpha value
alpha.fit <- tibble(alpha, RMSE)
alpha.min <- filter(alpha.fit, 
                    RMSE == min(RMSE))
 
# plot RMSE vs. alpha
ggplot(alpha.fit, aes(alpha, RMSE)) +
  geom_line() +
  geom_point(data = alpha.min,
             aes(alpha, RMSE), 
             size = 2, color = "red")
```

![](IntroToTimeSeries_files/figure-gfm/ses_validation-1.png)<!-- -->

The minimum RMSE was reached at
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") =
0.01, which suggests increasing the weights of more historical values
will better capture the trend of Google’s stock price.

Refit the forecast to the optimized
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") value
of 0.01.

``` r

# refit model with alpha = .01
ses.goog.opt <- ses(google.diff, 
                    alpha = .01,
                    h = 100)
 
# performance eval
accuracy(ses.goog.opt, goog.diff.test)
##                       ME     RMSE      MAE      MPE     MAPE      MASE
## Training set -0.01139031 8.829941 5.865211 99.32230 126.7409 0.6940567
## Test set     -0.00412292 8.083196 6.015444 92.14942 154.1342 0.7118344
##                    ACF1 Theil's U
## Training set 0.02886557        NA
## Test set     0.12278141  1.006324
 
# plotting results
p1 <- autoplot(ses.goog.opt) +
  theme(legend.position = "bottom")
p2 <- autoplot(goog.diff.test) +
  autolayer(ses.goog.opt, alpha = .5) +
  ggtitle("Predicted vs. actuals for
                 the test data set")
 
grid.arrange(p1, p2, 
                        nrow = 1)
```

![](IntroToTimeSeries_files/figure-gfm/SES_google_optimized-1.png)<!-- -->

# Holt’s method

With SES we had to remove the long-term trends to improve the model. But
in **Holt’s Method**, we can apply exponential smoothing while we are
capturing the time trend in the data. This is a technique that works
with data having a time trend but no seasonality. In order to make
predictions on the data, Holt’s Method uses **two smoothing parameters,
alpha, and beta**, which correspond to the level components and trend
components.

``` r

# applying holt's method on
# Google stock Data
holt.goog <- holt(google.train,
                  h = 100)

autoplot(holt.goog)
```

![](IntroToTimeSeries_files/figure-gfm/holt_google-1.png)<!-- -->

Holt’s method will automatically set the optimal value of
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha"), but
we can optionally set the parameters as well. Again,
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") close
to 1 indicates fast learning, while
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") close
to 0 indicates slow learning.

``` r

# holt's method
holt.goog$model
## Holt's method 
## 
## Call:
##  holt(y = google.train, h = 100) 
## 
##   Smoothing parameters:
##     alpha = 0.9999 
##     beta  = 1e-04 
## 
##   Initial states:
##     l = 401.1276 
##     b = 0.4091 
## 
##   sigma:  8.8149
## 
##      AIC     AICc      BIC 
## 10045.74 10045.81 10069.75
 
# accuracy of the model
accuracy(holt.goog, google.test)
##                        ME      RMSE       MAE         MPE     MAPE     MASE
## Training set -0.003332796  8.795267  5.821057 -0.01211821 1.000720 1.002452
## Test set      0.545744415 16.328680 12.876836  0.03013427 1.646261 2.217538
##                    ACF1 Theil's U
## Training set 0.03100836        NA
## Test set     0.87733298  2.024518
```

The parameter
![\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta") = 0.0001
was used to remove errors from the training set. We can further tune our
beta from this value.

We can optimize the value of
![\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta") through a
loop ranging from 0.0001 to 0.5 that will minimize the RMSE test.

``` r

# identify optimal beta parameter
beta <- seq(.0001, .5, by = .001)
RMSE <- NA
for(i in seq_along(beta)) {
  fit <- holt(google.train,
              beta = beta[i], 
              h = 100)
  RMSE[i] <- accuracy(fit, 
                      google.test)[2,2]
}
 
# convert to a data frame and
# identify min beta value
beta.fit <- data_frame(beta, RMSE)
beta.min <- filter(beta.fit, 
                   RMSE == min(RMSE))
 
# plot RMSE vs. beta
ggplot(beta.fit, aes(beta, RMSE)) +
  geom_line() +
  geom_point(data = beta.min, 
             aes(beta, RMSE), 
             size = 2, color = "red")
```

![](IntroToTimeSeries_files/figure-gfm/holt_google_optimize_beta-1.png)<!-- -->

The optimal
![\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta") value is
0.0231. Now refit the training model with our optimal
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") and
![\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta")
parameters and compare to the original model.

``` r

holt.goog <- holt(google.train,
                  h = 100)
 
# new model with optimal beta
holt.goog.opt <- holt(google.train,
                      h = 100,
                      beta = beta.min[[1]])
 
# accuracy of first model
accuracy(holt.goog, google.test)
##                        ME      RMSE       MAE         MPE     MAPE     MASE
## Training set -0.003332796  8.795267  5.821057 -0.01211821 1.000720 1.002452
## Test set      0.545744415 16.328680 12.876836  0.03013427 1.646261 2.217538
##                    ACF1 Theil's U
## Training set 0.03100836        NA
## Test set     0.87733298  2.024518
 
# accuracy of new optimal model
accuracy(holt.goog.opt, google.test)
##                       ME      RMSE       MAE          MPE     MAPE     MASE
## Training set -0.01109687  8.869111  5.928213 -0.007390802 1.018514 1.020906
## Test set      1.07170279 16.317372 12.915158  0.096552793 1.650023 2.224137
##                   ACF1 Theil's U
## Training set 0.0246560        NA
## Test set     0.8767228  2.020702
 
p1 <- autoplot(holt.goog) +
  ggtitle("Original Holt's Model") +
  coord_cartesian(ylim = c(400, 1000))
 
p2 <- autoplot(holt.goog.opt) +
  ggtitle("Optimal Holt's Model") +
  coord_cartesian(ylim = c(400, 1000))
 
grid.arrange(p1, p2,nrow = 1)
```

![](IntroToTimeSeries_files/figure-gfm/holt_google_compare_models-1.png)<!-- -->

The optimized model is more conservative, and the confidence interval is
more extreme.

# Holt-Winter’s Seasonal method

Google’s stock prices aren’t likely to have a major seasonal component,
but something like crop yield, or gas prices likely would have strong
seasonal components.

The Holt-Winter’s Seasonal method is used for data with both seasonal
patterns and time trends. This method can be implemented either by using
an *additive structure* or by using a *multiplicative structure*
depending on the data set. The additive structure or model is used when
the seasonal pattern of data has the same magnitude or is consistent
throughout, while the multiplicative structure or model is used if the
magnitude of the seasonal pattern of the data increases over time. It
uses three smoothing parameters,-
![\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha"),
![\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\beta"), and
![\gamma](https://latex.codecogs.com/png.latex?%5Cgamma "\gamma").

An additive model linearly decreases the weights of historic events,
while a multiplicative model exponentially decreases the weights of
historical events.

Use the decompose() function to perform this kind of exponential
smoothing. This example uses the qcement dataset.

The below example uses the Quarterly Australian Portland Cement
Production data.

``` r

# create training and validation
# of the qcement data
qcement.train <- window(qcement, 
                        end = c(2012, 4))
qcement.test <- window(qcement,
                       start = c(2013, 1))
 
# applying holt-winters 
# method on qcement
autoplot(decompose(qcement))
```

![](IntroToTimeSeries_files/figure-gfm/holtwinter_qcement-1.png)<!-- -->

Here, the seasonal component does not appear to increase with time,
suggesting an *additive model* will provide a better fit.

To create an Additive Model that deals with error, trend, and
seasonality, we are going to use the ets() function. Out of the 36
models, the ets() chooses the best additive model. For additive model,
the model parameter of ets() will be ‘AAA’.

``` r

# applying ets
qcement.hw <- ets(qcement.train, model = "AAA") # define additive

autoplot(forecast(qcement.hw))
```

![](IntroToTimeSeries_files/figure-gfm/ets_qcement-1.png)<!-- -->

Assess the model and summarize the smoothing parameters. Check the
residuals and find out the accuracy of our model.

``` r
 
# assessing our model
summary(qcement.hw)
## ETS(A,A,A) 
## 
## Call:
##  ets(y = qcement.train, model = "AAA") 
## 
##   Smoothing parameters:
##     alpha = 0.6418 
##     beta  = 1e-04 
##     gamma = 0.1988 
## 
##   Initial states:
##     l = 0.4511 
##     b = 0.0075 
##     s = 0.0049 0.0307 9e-04 -0.0365
## 
##   sigma:  0.0854
## 
##      AIC     AICc      BIC 
## 126.0419 126.8676 156.9060 
## 
## Training set error measures:
##                       ME       RMSE       MAE          MPE     MAPE      MASE
## Training set 0.001463693 0.08393279 0.0597683 -0.003454533 3.922727 0.5912949
##                    ACF1
## Training set 0.02150539
checkresiduals(qcement.hw)
```

![](IntroToTimeSeries_files/figure-gfm/ets_qcement_resids_accuracy-1.png)<!-- -->

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from ETS(A,A,A)
    ## Q* = 13.527, df = 8, p-value = 0.09495
    ## 
    ## Model df: 0.   Total lags used: 8

Residuals look good.

``` r
# forecast the next 5 quarters
qcement.f1 <- forecast(qcement.hw, h = 5)
 
# check accuracy
accuracy(qcement.f1, qcement.test)
##                       ME       RMSE        MAE          MPE     MAPE      MASE
## Training set 0.001463693 0.08393279 0.05976830 -0.003454533 3.922727 0.5912949
## Test set     0.031362775 0.07144211 0.06791904  1.115342984 2.899446 0.6719311
##                     ACF1 Theil's U
## Training set  0.02150539        NA
## Test set     -0.31290496 0.2112428
```

Accuracy is pretty good (RMSE = 0.0714421).

Here is an example of how the Multiplicative model works using ets().
For that purpose, the model parameter of ets() will be ‘MAM’. However,
we already know that the additive model will fit this time series more
appropriately.

``` r

# applying ets
qcement.hw2 <- ets(qcement.train, model = "MAM") # define multiplicative

checkresiduals(qcement.hw2)
```

![](IntroToTimeSeries_files/figure-gfm/etsMAM_qcement-1.png)<!-- -->

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from ETS(M,A,M)
    ## Q* = 16.002, df = 8, p-value = 0.04235
    ## 
    ## Model df: 0.   Total lags used: 8

## Optimize the Holt-Winter’s additive model

Next we can optimize the
![\gamma](https://latex.codecogs.com/png.latex?%5Cgamma "\gamma")
parameter in order to minimize the error rate. Then find the accuracy
and also plot the predictive values. Since the seasonal

``` r

# forecast the next 5 quarters
qcement.f1 <- forecast(qcement.hw,
                       h = 5)
 
# check accuracy
accuracy(qcement.f1, qcement.test)
##                       ME       RMSE        MAE          MPE     MAPE      MASE
## Training set 0.001463693 0.08393279 0.05976830 -0.003454533 3.922727 0.5912949
## Test set     0.031362775 0.07144211 0.06791904  1.115342984 2.899446 0.6719311
##                     ACF1 Theil's U
## Training set  0.02150539        NA
## Test set     -0.31290496 0.2112428
 
gamma <- seq(0.01, 0.85, 0.01)
RMSE <- NA
 
for(i in seq_along(gamma)) {
  hw.expo <- ets(qcement.train, 
                 "AAA", 
                 gamma = gamma[i])
  future <- forecast(hw.expo, 
                     h = 5)
  RMSE[i] = accuracy(future, 
                     qcement.test)[2,2]
}
 
error <- data_frame(gamma, RMSE)

minimum <- filter(error, 
                  RMSE == min(RMSE))

ggplot(error, aes(gamma, RMSE)) +
  geom_line() +
  geom_point(data = minimum, 
             color = "blue", size = 2) +
  ggtitle("gamma's impact on 
            forecast errors",
  subtitle = "gamma = 0.21 minimizes RMSE")
```

![](IntroToTimeSeries_files/figure-gfm/etsAAA_qcement_optimize-1.png)<!-- -->

``` r
 
# previous model with additive error, trend and seasonality
accuracy(qcement.f1, qcement.test)
##                       ME       RMSE        MAE          MPE     MAPE      MASE
## Training set 0.001463693 0.08393279 0.05976830 -0.003454533 3.922727 0.5912949
## Test set     0.031362775 0.07144211 0.06791904  1.115342984 2.899446 0.6719311
##                     ACF1 Theil's U
## Training set  0.02150539        NA
## Test set     -0.31290496 0.2112428
```

Optimizing
![\gamma](https://latex.codecogs.com/png.latex?%5Cgamma "\gamma") has
brought the RMSE from 0.0839328 down to 0.0714421.

    ##                        ME       RMSE        MAE        MPE     MAPE      MASE
    ## Training set -0.001312025 0.08377557 0.05905971 -0.2684606 3.834134 0.5842847
    ## Test set      0.033492771 0.07148708 0.06775269  1.2096488 2.881680 0.6702854
    ##                     ACF1 Theil's U
    ## Training set  0.04832198        NA
    ## Test set     -0.35877010 0.2202448
    ##         Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    ## 2013 Q1       2.134650 2.025352 2.243947 1.967494 2.301806
    ## 2013 Q2       2.427828 2.299602 2.556055 2.231723 2.623934
    ## 2013 Q3       2.601989 2.457284 2.746694 2.380681 2.823296
    ## 2013 Q4       2.505001 2.345506 2.664496 2.261075 2.748927
    ## 2014 Q1       2.171068 1.987914 2.354223 1.890958 2.451179

![](IntroToTimeSeries_files/figure-gfm/etsAAA_qcement_forecast_optimized-1.png)<!-- -->

# Damping model

The damping method uses the **damping coefficient**
![\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi") to estimate
more conservatively the predicted trends. The value of phi lies between
0 and 1. If we believe that our additive and multiplicative model is
going to be a flat line then chances are that it is damped. To
understand the working principle of damping forecasting we will use the
fpp2::ausair data set where we will create many models and try to have
much more conservative trend lines,

``` r

# Damping model in R
 
# holt's linear (additive) model
fit1 <- ets(ausair, model = "ZAN",
            alpha = 0.8, beta = 0.2)
pred1 <- forecast(fit1, h = 5)
 
# holt's linear (additive) model
fit2 <- ets(ausair, model = "ZAN", 
            damped = TRUE, alpha = 0.8, 
            beta = 0.2, phi = 0.85)
pred2 <- forecast(fit2, h = 5)
 
# holt's exponential
# (multiplicative) model
fit3 <- ets(ausair, model = "ZMN",
            alpha = 0.8, beta = 0.2)
pred3 <- forecast(fit3, h = 5)
 
# holt's exponential 
# (multiplicative) model damped
fit4 <- ets(ausair, model = "ZMN", 
            damped = TRUE,
            alpha = 0.8, beta = 0.2,
            phi = 0.85)
pred4 <- forecast(fit4, h = 5)
 
autoplot(ausair) +
  autolayer(pred1$mean, 
            color = "blue") +
  autolayer(pred2$mean, 
            color = "blue",
            linetype = "dashed") +
  autolayer(pred3$mean, 
            color = "red") +
  autolayer(pred4$mean, 
            color = "red", 
            linetype = "dashed")
```

![](IntroToTimeSeries_files/figure-gfm/damping_model-1.png)<!-- -->

Solid lines are have no
![\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi") parameter,
while dashed lines have
![\phi](https://latex.codecogs.com/png.latex?%5Cphi "\phi") = 0.85. The
blue lines are additive models and the red lines are multiplicative
models.

# Where to go Next

- Advanced time series models
  - ARCH, GARCH, etc. that model changing variance over time
- Vector Autoregression (VAR)
  - For multivariate i.e. multiple time series and modeling dependencies
    between them
- Machine Learning
  - How to do CV with time series
  - Neural networks for sequence data (LSTMs, etc.)
- Spatial Statistics
  - Generalize time dependence to spatial dependence in multiple
    dimensions
- Econometrics
  - Cointegration
  - Granger Causality
  - Serial correlation
  - Regression with time series data
- Bayesian time series

# Session Information

    R version 4.2.3 (2023-03-15 ucrt)
    Platform: x86_64-w64-mingw32/x64 (64-bit)
    Running under: Windows 10 x64 (build 19045)

    Matrix products: default

    locale:
    [1] LC_COLLATE=English_United States.utf8 
    [2] LC_CTYPE=English_United States.utf8   
    [3] LC_MONETARY=English_United States.utf8
    [4] LC_NUMERIC=C                          
    [5] LC_TIME=English_United States.utf8    

    attached base packages:
    [1] stats     graphics  grDevices utils     datasets  methods   base     

    other attached packages:
     [1] ggthemes_4.2.4  tseries_0.10-55 forecast_8.22.0 gridExtra_2.3  
     [5] scales_1.3.0    magrittr_2.0.3  IRdisplay_1.1   cowplot_1.1.1  
     [9] lubridate_1.9.2 forcats_1.0.0   stringr_1.5.0   dplyr_1.1.1    
    [13] purrr_1.0.1     readr_2.1.4     tidyr_1.3.0     tibble_3.2.1   
    [17] ggplot2_3.5.0   tidyverse_2.0.0

    loaded via a namespace (and not attached):
     [1] Rcpp_1.0.10       lattice_0.20-45   zoo_1.8-12        rprojroot_2.0.3  
     [5] digest_0.6.31     lmtest_0.9-40     utf8_1.2.3        R6_2.5.1         
     [9] repr_1.1.7        evaluate_0.20     highr_0.10        pillar_1.9.0     
    [13] rlang_1.1.0       curl_5.0.0        rstudioapi_0.14   fracdiff_1.5-3   
    [17] TTR_0.24.4        rmarkdown_2.21    labeling_0.4.2    munsell_0.5.0    
    [21] compiler_4.2.3    xfun_0.38         pkgconfig_2.0.3   base64enc_0.1-3  
    [25] urca_1.3-3        htmltools_0.5.5   nnet_7.3-18       tidyselect_1.2.0 
    [29] codetools_0.2-19  quadprog_1.5-8    fansi_1.0.4       tzdb_0.3.0       
    [33] withr_2.5.0       grid_4.2.3        nlme_3.1-162      jsonlite_1.8.4   
    [37] gtable_0.3.3      lifecycle_1.0.3   quantmod_0.4.26   cli_3.6.1        
    [41] stringi_1.7.12    farver_2.1.1      timeDate_4022.108 xts_0.13.1       
    [45] generics_0.1.3    vctrs_0.6.1       tools_4.2.3       glue_1.6.2       
    [49] hms_1.1.3         parallel_4.2.3    fastmap_1.1.1     yaml_2.3.7       
    [53] timechange_0.2.0  colorspace_2.1-0  knitr_1.42       
