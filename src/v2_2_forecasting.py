import numpy as np
import matplotlib.pyplot as plt

from utils import (
    generate_synthetic_gompertz_data,
    fit_all_models,
    make_predictions,
    calculate_metrics
)


# ============================================================
# CONFIGURATION
# ============================================================

NUMBER_OF_RUNS = 100
TRAINING_FRACTION = 0.50
NOISE_SCALE = 50

TRUE_V0 = 1
TRUE_K = 1000
TRUE_A = 0.05


# ============================================================
# VISUALIZATION
# ============================================================

def plot_average_performance(
    rmse_means,
    rmse_stds,
    mae_means,
    mae_stds,
    number_of_runs
):
    """
    Plot average RMSE and MAE across repeated forecasting runs.
    """

    models = ["Gompertz", "Logistic", "Exponential"]
    colors = ["steelblue", "darkorange", "darkgreen"]

    # Full RMSE comparison
    plt.figure(figsize=(8, 6))

    plt.bar(
        models,
        rmse_means,
        yerr=rmse_stds,
        capsize=6,
        color=colors
    )

    plt.xlabel("Tumor Growth Model")
    plt.ylabel("Mean RMSE")
    plt.title(
        f"Average Forecast RMSE Across {number_of_runs} Runs"
    )

    plt.tight_layout()
    plt.show()

    # Full MAE comparison
    plt.figure(figsize=(8, 6))

    plt.bar(
        models,
        mae_means,
        yerr=mae_stds,
        capsize=6,
        color=colors
    )

    plt.xlabel("Tumor Growth Model")
    plt.ylabel("Mean MAE")
    plt.title(
        f"Average Forecast MAE Across {number_of_runs} Runs"
    )

    plt.tight_layout()
    plt.show()

    # Zoomed RMSE comparison
    plt.figure(figsize=(8, 6))

    plt.bar(
        models[:2],
        rmse_means[:2],
        yerr=rmse_stds[:2],
        capsize=6,
        color=colors[:2]
    )

    plt.xlabel("Tumor Growth Model")
    plt.ylabel("Mean RMSE")
    plt.title("Forecast RMSE: Gompertz vs. Logistic")

    plt.tight_layout()
    plt.show()

    # Zoomed MAE comparison
    plt.figure(figsize=(8, 6))

    plt.bar(
        models[:2],
        mae_means[:2],
        yerr=mae_stds[:2],
        capsize=6,
        color=colors[:2]
    )

    plt.xlabel("Tumor Growth Model")
    plt.ylabel("Mean MAE")
    plt.title("Forecast MAE: Gompertz vs. Logistic")

    plt.tight_layout()
    plt.show()


# ============================================================
# PRINT RESULTS
# ============================================================

def print_summary(
    rmse_means,
    rmse_stds,
    mae_means,
    mae_stds,
    number_of_runs
):
    """
    Print the average forecasting metrics across all runs.
    """

    models = ["Gompertz", "Logistic", "Exponential"]

    print(
        f"\nAverage Forecast Performance Across "
        f"{number_of_runs} Runs"
    )

    print(
        f"{'Model':<14}"
        f"{'RMSE Mean':>12}"
        f"{'RMSE Std':>12}"
        f"{'MAE Mean':>12}"
        f"{'MAE Std':>12}"
    )

    print("-" * 62)

    for index, model in enumerate(models):
        print(
            f"{model:<14}"
            f"{rmse_means[index]:>12.2f}"
            f"{rmse_stds[index]:>12.2f}"
            f"{mae_means[index]:>12.2f}"
            f"{mae_stds[index]:>12.2f}"
        )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    time = np.linspace(0, 150, num=50)

    split_index = int(len(time) * TRAINING_FRACTION)

    # Each dictionary stores results for all three models.
    rmse_results = {
        "Gompertz": [],
        "Logistic": [],
        "Exponential": []
    }

    mae_results = {
        "Gompertz": [],
        "Logistic": [],
        "Exponential": []
    }

    for seed in range(NUMBER_OF_RUNS):
        _, observed_values = generate_synthetic_gompertz_data(
            time=time,
            v0=TRUE_V0,
            k=TRUE_K,
            a=TRUE_A,
            noise_scale=NOISE_SCALE,
            random_seed=seed
        )

        # Use the first 50% for training.
        train_time = time[:split_index]
        train_values = observed_values[:split_index]

        # Hold the remaining 50% out for testing.
        test_time = time[split_index:]
        test_values = observed_values[split_index:]

        try:
            gomp_params, exp_params, log_params = fit_all_models(
                train_time,
                train_values
            )

        except RuntimeError:
            print(
                f"Warning: Model fitting failed for seed {seed}. "
                "This run was skipped."
            )
            continue

        # Predict only on the unseen test timeline.
        predicted_gomp, predicted_exp, predicted_log = (
            make_predictions(
                test_time,
                gomp_params,
                exp_params,
                log_params
            )
        )

        predictions = {
            "Gompertz": predicted_gomp,
            "Logistic": predicted_log,
            "Exponential": predicted_exp
        }

        for model_name, predicted_values in predictions.items():
            rmse, mae = calculate_metrics(
                test_values,
                predicted_values
            )

            rmse_results[model_name].append(rmse)
            mae_results[model_name].append(mae)

    model_order = [
        "Gompertz",
        "Logistic",
        "Exponential"
    ]

    rmse_means = [
        np.mean(rmse_results[model])
        for model in model_order
    ]

    rmse_stds = [
        np.std(rmse_results[model])
        for model in model_order
    ]

    mae_means = [
        np.mean(mae_results[model])
        for model in model_order
    ]

    mae_stds = [
        np.std(mae_results[model])
        for model in model_order
    ]

    completed_runs = len(rmse_results["Gompertz"])

    print_summary(
        rmse_means,
        rmse_stds,
        mae_means,
        mae_stds,
        completed_runs
    )

    plot_average_performance(
        rmse_means,
        rmse_stds,
        mae_means,
        mae_stds,
        completed_runs
    )


if __name__ == "__main__":
    main()