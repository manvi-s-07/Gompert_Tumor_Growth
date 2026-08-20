import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from models import (
    gompertz_model,
    logistic_model,
    exponential_model,
    gompertz_for_fit,
    logistic_for_fit,
    exponential_for_fit
)


# =======================================================
# DATA GENERATION 
# =======================================================

def generate_synthetic_gompertz_data(
    time,
    v0,
    k,
    a,
    noise_scale=50,
    random_seed=42
):
    rng = np.random.default_rng(random_seed)

    true_values = gompertz_model(v0, k, a, time)

    noise = rng.normal(
        loc=0,
        scale=noise_scale,
        size=len(time)
    )

    observed_values = true_values + noise
    observed_values = np.maximum(observed_values, 0)

    return true_values, observed_values


# =======================================================
# MODEL FITTING
# =======================================================

def fit_all_models(time, observed_values):
    gomp_params, _ = curve_fit(gompertz_for_fit, time, observed_values, p0=[1, 1000, 0.05], 
                            bounds=(
                                [0.01, 1, 0.0001],
                                [500, 10000, 1]), maxfev=20000)
    
    exp_params, _ = curve_fit(exponential_for_fit, time, observed_values, p0=[1, 0.05],
                              bounds=(
                                [0.01, 0.0001],
                                [500, 1]), maxfev=20000)
    
    log_params, _ = curve_fit(logistic_for_fit, time, observed_values, p0=[1, 0.05, 1000],
                              bounds=(
                                [0.01, 0.0001, 1],
                                [500, 1, 10000]), maxfev=20000)

    #fitted_gomp = gompertz_model(gomp_params[0], gomp_params[1], gomp_params[2], time)
    #fitted_exp = exponential_model(exp_params[0], exp_params[1], time)
    #fitted_log = logistic_model(log_params[0], log_params[1], log_params[2], time)

    #return fitted_gomp, fitted_exp, fitted_log, gomp_params, exp_params, log_params
    return gomp_params, exp_params, log_params


# ============================================================
# PREDICTIONS
# ============================================================

def make_predictions(time, gomp_params, exp_params, log_params):
    """
    Use the fitted parameters to predict across any supplied timeline.
    """

    predicted_gomp = gompertz_model(
        gomp_params[0],
        gomp_params[1],
        gomp_params[2],
        time
    )

    predicted_exp = exponential_model(
        exp_params[0],
        exp_params[1],
        time
    )

    predicted_log = logistic_model(
        log_params[0],
        log_params[1],
        log_params[2],
        time
    )

    return predicted_gomp, predicted_exp, predicted_log





# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(observed_values, predictions):
    errors = observed_values - predictions

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    return rmse, mae

