import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =======================================================
# TUMOR GROWTH MODELS
# =======================================================

def exponential_model(v0, r, t):
    volume = v0 * np.exp((r*t))
    return volume 

def logistic_model(v0, r, k, t):
    volume = (v0 * k * np.exp((r * t)))/((k - v0) + (v0 * np.exp((r * t))))
    return volume

def gompertz_model(v0, k, a, t):
    volume = k * (np.exp(-np.log(k/v0) * np.exp(-a * t)))
    return volume



# Curve fits for parameters 

def gompertz_for_fit(t, v0, k, a):
    return gompertz_model(v0, k, a, t)

def logistic_for_fit(t, v0, r, k):
    return logistic_model(v0, r, k, t)

def exponential_for_fit(t, v0, r):
    return exponential_model(v0, r, t)


# =======================================================
# DATA GENERATION 
# =======================================================

def generate_synthetic_gompertz_data(time, v0, k, a, noise_scale=50, random_seed = 42):
    true_values = gompertz_model(v0, k, a, time)

    noise = np.random.normal(loc=0, scale=noise_scale, size=len(time))
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


def print_forecast_metrics(
    test_values,
    predicted_gomp_test,
    predicted_exp_test,
    predicted_log_test
):
    gomp_rmse, gomp_mae = calculate_metrics(
        test_values,
        predicted_gomp_test
    )

    exp_rmse, exp_mae = calculate_metrics(
        test_values,
        predicted_exp_test
    )

    log_rmse, log_mae = calculate_metrics(
        test_values,
        predicted_log_test
    )

    print("\nForecast Performance on Unseen Data:")
    print("Model        RMSE        MAE")
    print("-----------------------------------")
    print(f"Gompertz     {gomp_rmse:8.2f}   {gomp_mae:8.2f}")
    print(f"Logistic     {log_rmse:8.2f}   {log_mae:8.2f}")
    print(f"Exponential  {exp_rmse:8.2f}   {exp_mae:8.2f}")




# ============================================================
# VISUALIZATION
# ============================================================

def plot_forecasts(
    time,
    train_time,
    train_values,
    test_time,
    test_values,
    true_values,
    predicted_gomp,
    predicted_exp,
    predicted_log
):
    plt.figure(figsize=(10, 6))

    # Training and test observations
    plt.scatter(
        train_time,
        train_values,
        label="Training observations",
        marker="o"
    )

    plt.scatter(
        test_time,
        test_values,
        label="Unseen test observations",
        marker="x"
    )

    # Original curve used to generate the data
    plt.plot(
        time,
        true_values,
        label="True Gompertz growth",
        linestyle="--"
    )

    # Forecasts
    plt.plot(
        time,
        predicted_gomp,
        label="Gompertz forecast"
    )

    plt.plot(
        time,
        predicted_log,
        label="Logistic forecast"
    )

    plt.plot(
        time,
        predicted_exp,
        label="Exponential forecast"
    )

    # Marks where the unseen forecast region begins
    split_time = test_time[0]

    plt.axvline(
        x=split_time,
        linestyle=":",
        label="Forecast begins"
    )

    plt.xlabel("Time")
    plt.ylabel("Tumor volume")
    plt.title("Tumor Growth Forecasting from Early Observations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1400)

    plt.show()





# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    time = np.linspace(0, 150, num=50)

    true_v0 = 1
    true_k = 1000
    true_a = 0.05

    gomp_rmse_list = []
    log_rmse_list = []
    exp_rmse_list = []

    gomp_mae_list = []
    log_mae_list = []
    exp_mae_list = []

    
    
    true_gomp, observed_gomp = generate_synthetic_gompertz_data(
        time = time,
        v0 = true_v0,
        k = true_k,
        a = true_a,
        noise_scale = 50,
        random_seed=42
    )

    # Use the first 50% for training
    split_index = len(time) // 2

    train_time = time[:split_index]
    train_values = observed_gomp[:split_index]

    # Hold the remaining 50% out as unseen test data
    test_time = time[split_index:]
    test_values = observed_gomp[split_index:]

    # Fit using training data only
    gomp_params, exp_params, log_params = fit_all_models(
        train_time,
        train_values
    )

    # Predict across the full timeline
    predicted_gomp, predicted_exp, predicted_log = make_predictions(
        time,
        gomp_params,
        exp_params,
        log_params
    )

    # Extract predictions from the unseen test region
    predicted_gomp_test = predicted_gomp[split_index:]
    predicted_exp_test = predicted_exp[split_index:]
    predicted_log_test = predicted_log[split_index:]


    



    print("Training observations:", len(train_time))
    print("Testing observations:", len(test_time))

    print("\nFitted Gompertz parameters:")
    print(gomp_params)

    print("\nFitted Exponential parameters:")
    print(exp_params)

    print("\nFitted Logistic parameters:")
    print(log_params)

    print_forecast_metrics(
        test_values,
        predicted_gomp_test,
        predicted_exp_test,
        predicted_log_test
    )

    plot_forecasts(
        time,
        train_time,
        train_values,
        test_time,
        test_values,
        true_gomp,
        predicted_gomp,
        predicted_exp,
        predicted_log
    )