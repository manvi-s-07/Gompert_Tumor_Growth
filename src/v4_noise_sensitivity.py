import numpy as np
import matplotlib.pyplot as plt

from utils import (
    generate_synthetic_gompertz_data,
    fit_all_models,
    make_predictions,
    calculate_metrics
)


NUMBER_OF_RUNS = 100
TRAINING_FRACTION = 0.50

TRUE_V0 = 1
TRUE_K = 1000
TRUE_A = 0.05


NOISE_LEVELS = [
    0,
    10,
    25,
    50,
    75,
    100
]

def plot_noise_levels_results(
    noise_levels,
    gomp_rmse_means,
    log_rmse_means,
    exp_rmse_means,
    gomp_rmse_stds,
    log_rmse_stds,
    exp_rmse_stds,
    gomp_mae_means,
    log_mae_means,
    exp_mae_means,
    gomp_mae_stds,
    log_mae_stds,
    exp_mae_stds
):
  

    # ========================================================
    # RMSE - ALL MODELS
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        noise_levels,
        gomp_rmse_means,
        marker="o",
        label="Gompertz"
    )

    plt.plot(
        noise_levels,
        log_rmse_means,
        marker="o",
        label="Logistic"
    )

    plt.plot(
        noise_levels,
        exp_rmse_means,
        marker="o",
        label="Exponential"
    )

    plt.yscale("log")

    plt.xlabel("Measurement Noise (SD)")
    plt.ylabel("Mean Forecast RMSE (Log Scale)")
    plt.title("Effect of Measurment Noise on Forecast RMSE")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ========================================================
    # MAE - ALL MODELS
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        noise_levels,
        gomp_mae_means,
        marker="o",
        label="Gompertz"
    )

    plt.plot(
        noise_levels,
        log_mae_means,
        marker="o",
        label="Logistic"
    )

    plt.plot(
        noise_levels,
        exp_mae_means,
        marker="o",
        label="Exponential"
    )

    plt.yscale("log")

    plt.xlabel("Measurement Noise (SD)")
    plt.ylabel("Mean Forecast MAE (Log Scale)")
    plt.title("Effect of Measurement Noise on Forecast MAE")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ========================================================
    # RMSE - GOMPERTZ VS LOGISTIC
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        noise_levels,
        gomp_rmse_means,
        yerr=gomp_rmse_stds,
        marker="o",
        capsize=5,
        label="Gompertz"
    )

    plt.errorbar(
        noise_levels,
        log_rmse_means,
        yerr=log_rmse_stds,
        marker="o",
        capsize=5,
        label="Logistic"
    )

    plt.xlabel("Measurement Noise (SD)")
    plt.ylabel("Mean Forecast RMSE")
    plt.title(
        "Forecast RMSE vs. Measurement Noise"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ========================================================
    # MAE - GOMPERTZ VS LOGISTIC
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        noise_levels,
        gomp_mae_means,
        yerr=gomp_mae_stds,
        marker="o",
        capsize=5,
        label="Gompertz"
    )

    plt.errorbar(
        noise_levels,
        log_mae_means,
        yerr=log_mae_stds,
        marker="o",
        capsize=5,
        label="Logistic"
    )

    plt.xlabel("Measurement Noise (SD)")
    plt.ylabel("Mean Forecast MAE")
    plt.title(
        "Forecast MAE vs. Measurement Noise"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    time = np.linspace(0, 150, num=50)

    gomp_rmse_means = []
    log_rmse_means = []
    exp_rmse_means = []

    gomp_mae_means = []
    log_mae_means = []
    exp_mae_means = []

    
    gomp_rmse_stds = []
    log_rmse_stds = []
    exp_rmse_stds = []

    gomp_mae_stds = []
    log_mae_stds = []
    exp_mae_stds = []

    for noise_level in NOISE_LEVELS:
        gomp_rmse_runs = []
        log_rmse_runs = []
        exp_rmse_runs = []

        gomp_mae_runs = []
        log_mae_runs = []
        exp_mae_runs = []
        for seed in range(NUMBER_OF_RUNS):
            _, observed_gomp = generate_synthetic_gompertz_data(
                time=time,
                v0=TRUE_V0,
                k=TRUE_K,
                a=TRUE_A,
                noise_scale=noise_level,
                random_seed=seed
            )
            split_index = int(len(time) * TRAINING_FRACTION)

            train_time = time[:split_index]
            train_values = observed_gomp[:split_index]

            test_time = time[split_index:]
            test_values = observed_gomp[split_index:]
        
            gomp_params, exp_params, log_params = fit_all_models(
                train_time,
                train_values
            )

            predicted_gomp, predicted_exp, predicted_log = make_predictions(
                test_time,
                gomp_params,
                exp_params,
                log_params
            )

            gomp_rmse, gomp_mae = calculate_metrics(
                test_values,
                predicted_gomp
            )

            log_rmse, log_mae = calculate_metrics(
                test_values,
                predicted_log
            )

            exp_rmse, exp_mae = calculate_metrics(
                test_values,
                predicted_exp
            )

            gomp_rmse_runs.append(gomp_rmse)
            log_rmse_runs.append(log_rmse)
            exp_rmse_runs.append(exp_rmse)

            gomp_mae_runs.append(gomp_mae)
            log_mae_runs.append(log_mae)
            exp_mae_runs.append(exp_mae)

        gomp_rmse_means.append(np.mean(gomp_rmse_runs))
        log_rmse_means.append(np.mean(log_rmse_runs))
        exp_rmse_means.append(np.mean(exp_rmse_runs))

        gomp_mae_means.append(np.mean(gomp_mae_runs))
        log_mae_means.append(np.mean(log_mae_runs))
        exp_mae_means.append(np.mean(exp_mae_runs))

        gomp_rmse_stds.append(np.std(gomp_rmse_runs))
        log_rmse_stds.append(np.std(log_rmse_runs))
        exp_rmse_stds.append(np.std(exp_rmse_runs))

        gomp_mae_stds.append(np.std(gomp_mae_runs))
        log_mae_stds.append(np.std(log_mae_runs))
        exp_mae_stds.append(np.std(exp_mae_runs))
        

    print("Gompertz RMSE: ", gomp_rmse_means)
    print("Logistic RMSE: ", log_rmse_means)
    print("Exponential RMSE: ", exp_rmse_means)

    print("Gompertz MAE:", gomp_mae_means)
    print("Logistic MAE:", log_mae_means)
    print("Exponential MAE:", exp_mae_means)

    print("Gompertz RMSE SD:", gomp_rmse_stds)
    print("Logistic RMSE SD:", log_rmse_stds)
    print("Exponential RMSE SD:", exp_rmse_stds)

    print("Gompertz MAE SD:", gomp_mae_stds)
    print("Logistic MAE SD:", log_mae_stds)
    print("Exponential MAE SD:", exp_mae_stds)

    plot_noise_levels_results(
    NOISE_LEVELS,

    gomp_rmse_means,
    log_rmse_means,
    exp_rmse_means,

    gomp_rmse_stds,
    log_rmse_stds,
    exp_rmse_stds,

    gomp_mae_means,
    log_mae_means,
    exp_mae_means,

    gomp_mae_stds,
    log_mae_stds,
    exp_mae_stds
)

if __name__ == "__main__":
    main()


        

