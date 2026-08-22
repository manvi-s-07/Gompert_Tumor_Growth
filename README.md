# Tumor Growth Forecasting Research

## Overview

This project started from my previous work with the Gompertz growth model and has grown into a larger project looking at tumor growth forecasting.

I am currently comparing different mathematical growth models and testing how well they can predict future tumor growth when only earlier measurements are available.

The three models I am currently working with are:

- Gompertz
- Logistic
- Exponential

Instead of only looking at how well each model fits a tumor growth curve, I wanted to focus more on forecasting. In real situations, we would not already know the entire future growth trajectory, so I wanted to see what happens when the models only have access to earlier observations and have to predict what happens next.

For now, I am using synthetic tumor growth data generated from the Gompertz model. This lets me control things like measurement noise and how much data is available while still knowing the true underlying growth behavior.

My overall research question is:

> **Can machine learning improve tumor growth forecasting compared to classical mathematical growth models, especially when the available data is limited or noisy?**

The current part of the project is focused on building and understanding the classical-model baseline before moving into machine learning.


## Models

### Gompertz Model

The Gompertz model represents growth that starts relatively quickly and gradually slows as it approaches a maximum size.

This is currently the main model used to generate the synthetic tumor growth data in my experiments.


### Logistic Model

The Logistic model also represents growth that eventually approaches a maximum size, but its growth pattern is different from Gompertz growth.


### Exponential Model

The Exponential model assumes continued growth without a carrying capacity.

Including it gives me a simpler baseline and also helps show what happens when a model does not account for growth saturation.


## Project Progress

### V1 — Model Fitting

The first part of the project focused on making sure I could generate synthetic tumor growth data and fit different mathematical models to it.

I generated data using the Gompertz model, added Gaussian measurement noise, and then fit Gompertz, Logistic, and Exponential models to the same observations.

I compared their performance using RMSE and MAE.

This gave me the basic framework that I use throughout the later experiments.


### V2 — Early-Data Forecasting

The next step was moving from fitting to actual forecasting.

Instead of giving the models the entire tumor growth trajectory, I only gave them the first 50% of the observations.

The models were fit using this early data and then used to predict the remaining unseen portion of the tumor growth.

This helped me start looking at the difference between:

**"How well can a model fit data it has already seen?"**

and

**"How well can a model predict what happens next?"**


### V2.1 — Repeated Forecasting

One noisy dataset can give a result that happened partly because of random chance, so I expanded the forecasting experiment to run across 100 different noisy datasets.

For each run, I generated a new noisy tumor trajectory, trained the models on the early observations, and evaluated their forecasts on the unseen observations.

I then calculated the average RMSE and MAE along with their standard deviations.

This gave me a better idea of both the average forecasting performance and how consistent each model was.


### V3 — Effect of Available Data

For V3, I wanted to see how forecasting changes depending on how much of the tumor trajectory has already been observed.

I tested training fractions of:

- 20%
- 30%
- 40%
- 50%
- 60%
- 70%

Each condition was tested across 100 noisy datasets.

The main pattern was that having more tumor measurements greatly improved forecasting accuracy and stability.

With very limited early data, Gompertz and Logistic forecasts could both be highly unstable. Their performance improved a lot once more of the trajectory became available.

The Exponential model had especially large forecasting errors when only early observations were available because it continued predicting unrestricted growth.


### V4 — Effect of Measurement Noise

For V4, I kept the amount of training data fixed at 50% and changed the amount of measurement noise.

I tested noise standard deviations of:

- 0
- 10
- 25
- 50
- 75
- 100

Each noise level was again tested across 100 runs.

As expected, the Gompertz model performed extremely well when there was little or no noise. As the measurements became noisier, both its average forecasting error and the variability between runs increased.

One result I found interesting was that at the highest noise levels, the Logistic model had lower average forecasting error than Gompertz, even though the original synthetic data was generated using the Gompertz model.

This is something I want to investigate more later rather than making a general conclusion from the current experiment.


## Evaluation

I am currently using two main error metrics:

### RMSE

Root Mean Squared Error measures the difference between the predicted and observed tumor volumes and gives larger errors more weight.

### MAE

Mean Absolute Error measures the average absolute difference between the predictions and observations.

I also calculate the standard deviation across repeated experiments to see how stable or unstable the forecasting results are.


## Project Structure

```text
Gompertz_Tumor_Growth/
│
├── src/
│   ├── models.py
│   ├── utils.py
│   ├── v2_1_forecasting.py
│   ├── v2_2_forecasting.py
│   ├── v3_available_data.py
│   └── v4_noise_sensitivity.py
│
├── results/
│   ├── v2/
│   ├── v3/
│   └── v4/
│
└── README.md
```

### `models.py`

Contains the mathematical equations for the Gompertz, Logistic, and Exponential growth models.

### `utils.py`

Contains functions that are reused across experiments, including:

- synthetic data generation
- model fitting
- predictions
- RMSE and MAE calculations

The individual version files contain the code for each experiment.


## What I Have Found So Far

Some of the main things I have seen from the project so far are:

- Fitting existing tumor measurements is much easier than forecasting unseen growth.
- The amount of available data has a large effect on forecasting accuracy.
- Very early forecasts can be extremely unstable.
- Increasing measurement noise generally makes forecasting less accurate and less consistent.
- Gompertz and Logistic can behave differently depending on how much data or noise is present.
- A model matching the equation that generated the data does not necessarily give the lowest forecasting error under every condition.
- Exponential growth is not a good long-term forecasting model for this simulated saturating tumor growth.


## Where I Am Going Next

The next major part of the project is moving into machine learning.

So far, the classical models have given me a baseline and have also shown me some of the situations where tumor forecasting becomes difficult.

The next question I want to investigate is:

> **Can a machine learning model make more accurate or more stable forecasts than these classical growth models?**

I especially want to compare the approaches when:

- only early tumor measurements are available
- measurements contain more noise
- long-range forecasting is required

After building the ML portion, I plan to revisit some of the earlier experiments with a more complete comparison.

I also eventually want to move beyond synthetic data and test the forecasting methods using real longitudinal tumor growth datasets.


## Tools

This project currently uses:

- Python
- NumPy
- SciPy
- Matplotlib

Machine learning libraries will be added as that part of the project develops.


## Current Status

This project is still in progress.

The current experiments use controlled synthetic Gompertz data, so the results are meant to help develop and test the forecasting framework. They are not meant to make conclusions about real patient tumor growth yet.

The next stage of the project will focus on introducing machine learning models and comparing them against the classical forecasting results.
