from read_data import PremierLeagueDataProcessor
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data = PremierLeagueDataProcessor(start=True)

X_map = data.X
X = data.X_norm
y_all = data.y
y_folds = data.y_splits
y_zero_vs_rest = data.Y_zero_vs_rest
y_one_vs_rest = data.Y_one_vs_rest
y_two_vs_rest = data.Y_two_vs_rest
beta_init = data.beta
#print(len(beta))
"""print((X_train.iloc[0]))"""


def sigmoid(x):
    
    
    #print(1.0 / (1.0 + np.exp(-x)))
    return 1.0 / (1.0 + np.exp(-x))

def soft_threshold(rho, gamma):
    if rho > gamma:
        return rho - gamma
    elif rho < -gamma:
        return rho + gamma
    else:
        return 0


def sigmoid_matrix(z):
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))


def binary_logistic_loss(X, y, beta, regularizer="none", lmbda=0.0, eps=1e-12):
    X_mat = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    beta_arr = np.asarray(beta, dtype=float).reshape(-1)

    probs = sigmoid_matrix(X_mat @ beta_arr)
    probs = np.clip(probs, eps, 1.0 - eps)

    base_loss = -np.mean(y_arr * np.log(probs) + (1.0 - y_arr) * np.log(1.0 - probs))

    if regularizer == "lasso":
        penalty = lmbda * np.sum(np.abs(beta_arr[1:]))
    elif regularizer == "ridge":
        penalty = 0.5 * lmbda * np.sum(beta_arr[1:] ** 2)
    else:
        penalty = 0.0

    return float(base_loss + penalty)



def ridge_logistic_regression(
    X_folds,
    y_folds,
    beta_init,
    lmbda=0.1,
    learning_rate=0.01,
    iterations=1000,
    tol=1e-6,
    return_histories=False
):
    
    num_folds = len(X_folds)
    

    beta_init = np.asarray(beta_init, dtype=float).reshape(-1)
    betas = []
    loss_histories = []

    for leave_out in range(num_folds):
        X_train = np.concatenate([
            X_folds[i].to_numpy(dtype=float) if hasattr(X_folds[i], "to_numpy") else np.asarray(X_folds[i], dtype=float)
            for i in range(num_folds) if i != leave_out
        ], axis=0)

        y_train = np.concatenate([
            np.asarray(y_folds[i], dtype=float).reshape(-1)
            for i in range(num_folds) if i != leave_out
        ], axis=0)

        n_samples, n_features = X_train.shape

        

        beta = beta_init.copy()
        fold_losses = []


        if return_histories:
            fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="ridge", lmbda=lmbda))
        for _ in range(iterations):
            linear_pred = X_train @ beta
            predictions = sigmoid_matrix(linear_pred)

            error = predictions - y_train
            gradient = (X_train.T @ error) / n_samples

            penalty = lmbda * beta
            penalty[0] = 0.0   
            gradient += penalty

            beta_new = beta - learning_rate * gradient

            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="ridge", lmbda=lmbda))
                break

            beta = beta_new
            if return_histories:
                fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="ridge", lmbda=lmbda))

        betas.append(beta.copy())
        if return_histories:
            loss_histories.append(fold_losses)

    if return_histories:
        return betas, loss_histories
    return betas

def standard_logistic_regression(
    X_folds,
    y_folds,
    beta_init,
    learning_rate=0.01,
    iterations=1000,
    tol=1e-6,
    return_histories=False
):
    
    num_folds = len(X_folds)
    

    beta_init = np.asarray(beta_init, dtype=float).reshape(-1)
    betas = []
    loss_histories = []

    for leave_out in range(num_folds):
        X_train = np.concatenate([
            X_folds[i].to_numpy(dtype=float) if hasattr(X_folds[i], "to_numpy") else np.asarray(X_folds[i], dtype=float)
            for i in range(num_folds) if i != leave_out
        ], axis=0)

        y_train = np.concatenate([
            np.asarray(y_folds[i], dtype=float).reshape(-1)
            for i in range(num_folds) if i != leave_out
        ], axis=0)

        n_samples, n_features = X_train.shape

        

        beta = beta_init.copy()
        fold_losses = []
        if return_histories:
            fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="none", lmbda=0.0))
        for _ in range(iterations):
            linear_pred = X_train @ beta
            predictions = sigmoid_matrix(linear_pred)

            error = predictions - y_train
            gradient = (X_train.T @ error) / n_samples

            beta_new = beta - learning_rate * gradient

            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                if return_histories:
                    fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="none", lmbda=0.0))
                break

            beta = beta_new
            if return_histories:
                fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="none", lmbda=0.0))

        betas.append(beta.copy())
        if return_histories:
            loss_histories.append(fold_losses)

    if return_histories:
        return betas, loss_histories
    return betas


def lasso_logistic_regression(
    X_input,
    y_input,
    beta_init,
    lmbda,
    iterations=500,
    tol=1e-9,
    num_folds=8,
    return_histories=False
):
   
    if isinstance(X_input, (list, tuple)):
        

        X_folds = [
            x.to_numpy(dtype=float) if hasattr(x, "to_numpy") else np.asarray(x, dtype=float)
            for x in X_input
        ]
        y_folds = [
            np.asarray(y, dtype=float).reshape(-1)
            for y in y_input
        ]
    else:
        X_array = X_input.to_numpy(dtype=float) if hasattr(X_input, "to_numpy") else np.asarray(X_input, dtype=float)
        y_array = np.asarray(y_input, dtype=float).reshape(-1)

        

        X_folds = np.array_split(X_array, num_folds, axis=0)
        y_folds = np.array_split(y_array, num_folds, axis=0)

    beta_init = np.asarray(beta_init, dtype=float).reshape(-1)
    p = X_folds[0].shape[1]

   

    betas = []
    loss_histories = []

    for leave_out in range(num_folds):
        train_ix = [k for k in range(num_folds) if k != leave_out]

        X_train = np.concatenate([X_folds[k] for k in train_ix], axis=0)
        y_train = np.concatenate([y_folds[k] for k in train_ix], axis=0).reshape(-1)

        n, p_train = X_train.shape
        

        beta = beta_init.copy()
        fold_losses = []
        if return_histories:
            fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="lasso", lmbda=lmbda))

        for _ in range(iterations):
            beta_old = beta.copy()

            linear_preds = X_train @ beta
            probs = 1.0 / (1.0 + np.exp(-linear_preds))
            w = np.maximum(probs * (1.0 - probs), 1e-8)
            z = linear_preds + (y_train - probs) / w

            for j in range(p):
                gamma = 0.0 if j == 0 else n * lmbda

                partial = linear_preds - X_train[:, j] * beta[j]
                rj = np.sum((w * X_train[:, j]) * (z - partial))
                denom = np.sum(w * (X_train[:, j] ** 2)) + 1e-12

                if j == 0:
                    beta_j_new = rj / denom
                else:
                    beta_j_new = np.sign(rj) * max(abs(rj) - gamma, 0.0) / denom

                delta = beta_j_new - beta[j]
                if delta != 0.0:
                    linear_preds += X_train[:, j] * delta
                    beta[j] = beta_j_new

            if np.linalg.norm(beta - beta_old) < tol:
                if return_histories:
                    fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="lasso", lmbda=lmbda))
                break

            if return_histories:
                fold_losses.append(binary_logistic_loss(X_train, y_train, beta, regularizer="lasso", lmbda=lmbda))

        betas.append(beta.copy())
        if return_histories:
            loss_histories.append(fold_losses)

    if return_histories:
        return betas, loss_histories
    return betas


def test(X, beta0,beta1, beta2,y):
    X_mat = X.to_numpy(dtype=float) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=float)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(1, -1)

    y_arr = np.asarray(y).reshape(-1).astype(int)
    

    result_0 = sigmoid_matrix(X_mat @ beta0)
    result_1 = sigmoid_matrix(X_mat @ beta1)
    result_2 = sigmoid_matrix(X_mat @ beta2)
    predictions = np.argmax(np.column_stack([result_0, result_1, result_2]), axis=1)
    return float(np.mean(predictions == y_arr))


def evaluate_fold_models(X_folds, y_folds, beta0_list, beta1_list, beta2_list):
    num_folds = len(X_folds)
    

    success_rates = []
    for fold_idx in range(num_folds):
        fold_score = test(
            X_folds[fold_idx],
            beta0_list[fold_idx],
            beta1_list[fold_idx],
            beta2_list[fold_idx],
            y_folds[fold_idx]
        )
        success_rates.append(fold_score)
    return success_rates


def t_critical_95(df):
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
    }
    if df <= 0:
        return 0.0
    if df in t_table:
        return t_table[df]
    return 1.96


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))

    if n < 2:
        return {
            "mean": mean,
            "std": 0.0,
            "sem": 0.0,
            "half_width": 0.0,
            "lower": mean,
            "upper": mean,
            "confidence": confidence,
            "n": n,
        }

    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    if np.isclose(confidence, 0.95):
        critical = t_critical_95(n - 1)
    else:
        critical = 1.96

    half_width = critical * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "confidence": confidence,
        "n": n,
    }


def print_confidence_intervals(model_scores, confidence=0.95):
    confidence_pct = int(confidence * 100)
    print(f"\n=== {confidence_pct}% CONFIDENCE INTERVALS ===")
    for model_name, scores in model_scores.items():
        stats = confidence_interval(scores, confidence=confidence)
        print(
            f"{model_name}: mean={stats['mean']:.4f}, "
            f"CI=[{stats['lower']:.4f}, {stats['upper']:.4f}], "
            f"half-width={stats['half_width']:.4f}, n={stats['n']}"
        )


def plot_confidence_intervals(model_scores, confidence=0.95, save_path="confidence_intervals.png", show_plot=True):
    confidence_pct = int(confidence * 100)
    model_names = list(model_scores.keys())
    stats_list = [confidence_interval(model_scores[name], confidence=confidence) for name in model_names]
    means = [item["mean"] for item in stats_list]
    half_widths = [item["half_width"] for item in stats_list]

    x = np.arange(len(model_names))
    colors = ["#d95f02", "#1b9e77", "#7570b3"]

    plt.figure(figsize=(9, 6))
    plt.bar(x, means, yerr=half_widths, capsize=10, color=colors[:len(model_names)], alpha=0.8)

    for idx, name in enumerate(model_names):
        fold_scores = np.asarray(model_scores[name], dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(fold_scores)) if len(fold_scores) > 1 else np.array([0.0])
        plt.scatter(np.full(len(fold_scores), idx) + jitter, fold_scores, color="black", s=35, zorder=3)

    plt.xticks(x, model_names)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"Model Accuracy with {confidence_pct}% Confidence Intervals")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def train_ovr_regression(regression_name, X_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init, lmbda=0.0):
    if regression_name == "lasso":
        beta0 = lasso_logistic_regression(X_folds, y_zero_folds, beta_init, lmbda=lmbda)
        beta1 = lasso_logistic_regression(X_folds, y_one_folds, beta_init, lmbda=lmbda)
        beta2 = lasso_logistic_regression(X_folds, y_two_folds, beta_init, lmbda=lmbda)
    elif regression_name == "ridge":
        beta0 = ridge_logistic_regression(X_folds, y_zero_folds, beta_init, lmbda=lmbda)
        beta1 = ridge_logistic_regression(X_folds, y_one_folds, beta_init, lmbda=lmbda)
        beta2 = ridge_logistic_regression(X_folds, y_two_folds, beta_init, lmbda=lmbda)
    elif regression_name == "standard":
        beta0 = standard_logistic_regression(X_folds, y_zero_folds, beta_init)
        beta1 = standard_logistic_regression(X_folds, y_one_folds, beta_init)
        beta2 = standard_logistic_regression(X_folds, y_two_folds, beta_init)
    else:
        raise ValueError(f"Unknown regression name: {regression_name}")

    return beta0, beta1, beta2


def mean_accuracy_for_model(regression_name, X_folds, y_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init, lmbda=0.0):
    beta0, beta1, beta2 = train_ovr_regression(
        regression_name,
        X_folds,
        y_zero_folds,
        y_one_folds,
        y_two_folds,
        beta_init,
        lmbda=lmbda
    )
    fold_scores = evaluate_fold_models(X_folds, y_folds, beta0, beta1, beta2)
    return float(np.mean(fold_scores)), fold_scores


def lambda_sweep_accuracy(X_folds, y_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init, lambda_values):
    lasso_scores = []
    ridge_scores = []

    standard_mean, standard_fold_scores = mean_accuracy_for_model(
        "standard", X_folds, y_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init
    )
    standard_scores = [standard_mean] * len(lambda_values)

    for idx, lmbda in enumerate(lambda_values):
        lasso_mean, _ = mean_accuracy_for_model(
            "lasso", X_folds, y_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init, lmbda=lmbda
        )
        ridge_mean, _ = mean_accuracy_for_model(
            "ridge", X_folds, y_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init, lmbda=lmbda
        )
        lasso_scores.append(lasso_mean)
        ridge_scores.append(ridge_mean)
        

    return {
        "lambda_values": np.asarray(lambda_values, dtype=float),
        "lasso": np.asarray(lasso_scores, dtype=float),
        "ridge": np.asarray(ridge_scores, dtype=float),
        "standard": np.asarray(standard_scores, dtype=float),
        "standard_fold_scores": np.asarray(standard_fold_scores, dtype=float),
    }


def plot_lambda_accuracy_curves(sweep_results, save_path="lambda_accuracy_curve.png", show_plot=True):
    lambda_values = sweep_results["lambda_values"]
    lasso_scores = sweep_results["lasso"]
    ridge_scores = sweep_results["ridge"]
    standard_scores = sweep_results["standard"]

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, lasso_scores, label="Lasso Logistic", linewidth=2.2, color="#d95f02")
    plt.plot(lambda_values, ridge_scores, label="Ridge Logistic", linewidth=2.2, color="#1b9e77")
    plt.plot(lambda_values, standard_scores, label="Standard Logistic", linewidth=2.2, linestyle="--", color="#7570b3")

    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Lambda vs Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def pad_loss_history(loss_values, target_length):
    loss_values = list(loss_values)
    if len(loss_values) == 0:
        return np.full(target_length, np.nan, dtype=float)
    if len(loss_values) >= target_length:
        return np.asarray(loss_values[:target_length], dtype=float)
    padded = loss_values + [loss_values[-1]] * (target_length - len(loss_values))
    return np.asarray(padded, dtype=float)


def mean_ovr_loss_curve(class_histories_for_fold):
    max_len = max(len(history) for history in class_histories_for_fold)
    padded = np.vstack([pad_loss_history(history, max_len) for history in class_histories_for_fold])
    return np.nanmean(padded, axis=0)


def build_loss_history_summary(X_folds, y_zero_folds, y_one_folds, y_two_folds, beta_init, fold_idx=0):
    _, lasso_h0 = lasso_logistic_regression(X_folds, y_zero_folds, beta_init, lmbda=0.001, return_histories=True)
    _, lasso_h1 = lasso_logistic_regression(X_folds, y_one_folds, beta_init, lmbda=0.01, return_histories=True)
    _, lasso_h2 = lasso_logistic_regression(X_folds, y_two_folds, beta_init, lmbda=0.001, return_histories=True)

    _, ridge_h0 = ridge_logistic_regression(X_folds, y_zero_folds, beta_init, lmbda=0.001, return_histories=True)
    _, ridge_h1 = ridge_logistic_regression(X_folds, y_one_folds, beta_init, lmbda=0.01, return_histories=True)
    _, ridge_h2 = ridge_logistic_regression(X_folds, y_two_folds, beta_init, lmbda=0.001, return_histories=True)

    _, standard_h0 = standard_logistic_regression(X_folds, y_zero_folds, beta_init, return_histories=True)
    _, standard_h1 = standard_logistic_regression(X_folds, y_one_folds, beta_init, return_histories=True)
    _, standard_h2 = standard_logistic_regression(X_folds, y_two_folds, beta_init, return_histories=True)

    return {
        "Lasso": mean_ovr_loss_curve([lasso_h0[fold_idx], lasso_h1[fold_idx], lasso_h2[fold_idx]]),
        "Ridge": mean_ovr_loss_curve([ridge_h0[fold_idx], ridge_h1[fold_idx], ridge_h2[fold_idx]]),
        "Standard": mean_ovr_loss_curve([standard_h0[fold_idx], standard_h1[fold_idx], standard_h2[fold_idx]]),
    }


def print_loss_history_summary(loss_history_summary, fold_idx=0):
    print(f"\n=== LOSS VS ITERATION SUMMARY (Fold {fold_idx}) ===")
    for model_name, curve in loss_history_summary.items():
        print(
            f"{model_name}: iterations={len(curve)}, "
            f"initial_loss={curve[0]:.6f}, final_loss={curve[-1]:.6f}"
        )


def plot_loss_vs_iteration(loss_history_summary, fold_idx=0, save_path="loss_vs_iteration.png", show_plot=True):
    plt.figure(figsize=(10, 6))
    colors = {"Lasso": "#d95f02", "Ridge": "#1b9e77", "Standard": "#7570b3"}

    for model_name, curve in loss_history_summary.items():
        iterations = np.arange(1, len(curve) + 1)
        plt.plot(iterations, curve, label=model_name, linewidth=2.2, color=colors.get(model_name))

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Iteration (Average OvR Loss, Fold {fold_idx})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def correlation_map(X):
    X_mat = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X, dtype=float)

    
    X_sub = X_mat[:, 1:]

    
    average = np.mean(X_sub, axis=0)
    std = np.std(X_sub, axis=0, ddof=1)

   
    cov = np.cov(X_sub, rowvar=False, ddof=1)

    return average, std, cov
    
def plot_covariance_heatmap(X, labels=None, figsize=(14, 12), cmap='coolwarm', annot=False):
    
    if plt is None or sns is None:
        raise ImportError("plot_covariance_heatmap için matplotlib ve seaborn gerekli")

    _, _, cov = correlation_map(X)

    if labels is None:
        p = cov.shape[0]
        labels = [f"x{i}" for i in range(p)]

    plt.figure(figsize=figsize)
    sns.heatmap(cov, xticklabels=labels, yticklabels=labels,
                cmap=cmap, center=0, annot=annot, fmt=".2f",
                cbar_kws={"shrink": 0.8})
    plt.title("Covariance Matrix Heatmap (Bessel ddof=1)")
    plt.tight_layout()
    plt.show()





beta_zero_vs_rest_lasso = lasso_logistic_regression(X, y_zero_vs_rest, beta_init, lmbda=0.001)
beta_one_vs_rest_lasso = lasso_logistic_regression(X, y_one_vs_rest, beta_init, lmbda=0.01)
beta_two_vs_rest_lasso = lasso_logistic_regression(X, y_two_vs_rest, beta_init, lmbda=0.001)

succes_for_lasso = evaluate_fold_models(
    X, y_folds,
    beta_zero_vs_rest_lasso,
    beta_one_vs_rest_lasso,
    beta_two_vs_rest_lasso
)

beta_zero_vs_rest_standard = standard_logistic_regression(X, y_zero_vs_rest, beta_init)
beta_one_vs_rest_standard = standard_logistic_regression(X, y_one_vs_rest, beta_init)
beta_two_vs_rest_standard = standard_logistic_regression(X, y_two_vs_rest, beta_init)

succes_for_standard = evaluate_fold_models(
    X, y_folds,
    beta_zero_vs_rest_standard,
    beta_one_vs_rest_standard,
    beta_two_vs_rest_standard
)

beta_zero_vs_rest_ridge = ridge_logistic_regression(X, y_zero_vs_rest, beta_init, lmbda=0.001)
beta_one_vs_rest_ridge = ridge_logistic_regression(X, y_one_vs_rest, beta_init, lmbda=0.01)
beta_two_vs_rest_ridge = ridge_logistic_regression(X, y_two_vs_rest, beta_init, lmbda=0.001)

succes_for_ridge = evaluate_fold_models(
    X, y_folds,
    beta_zero_vs_rest_ridge,
    beta_one_vs_rest_ridge,
    beta_two_vs_rest_ridge
)

for i in range(len(succes_for_lasso)):
    print(
        f"Fold {i}: "
        f"Lasso Logit: {succes_for_lasso[i]:.4f}, "
        f"Ridge Logit: {succes_for_ridge[i]:.4f}, "
        f"Standard Logit: {succes_for_standard[i]:.4f}"
    )

print(
    f"\nAverage Accuracy -> "
    f"Lasso: {np.mean(succes_for_lasso):.4f}, "
    f"Ridge: {np.mean(succes_for_ridge):.4f}, "
    f"Standard: {np.mean(succes_for_standard):.4f}"
)

model_scores = {
    "Lasso": succes_for_lasso,
    "Ridge": succes_for_ridge,
    "Standard": succes_for_standard,
}

print_confidence_intervals(model_scores, confidence=0.95)
plot_confidence_intervals(model_scores, confidence=0.95)

selected_fold_for_loss_plot = 0
loss_history_summary = build_loss_history_summary(
    X,
    y_zero_vs_rest,
    y_one_vs_rest,
    y_two_vs_rest,
    beta_init,
    fold_idx=selected_fold_for_loss_plot
)
print_loss_history_summary(loss_history_summary, fold_idx=selected_fold_for_loss_plot)
plot_loss_vs_iteration(
    loss_history_summary,
    fold_idx=selected_fold_for_loss_plot,
    save_path="loss_vs_iteration.png",
    show_plot=True
)

max_loss_len = max(len(curve) for curve in loss_history_summary.values())
loss_curve_df = np.column_stack([
    np.arange(1, max_loss_len + 1),
    pad_loss_history(loss_history_summary["Lasso"], max_loss_len),
    pad_loss_history(loss_history_summary["Ridge"], max_loss_len),
    pad_loss_history(loss_history_summary["Standard"], max_loss_len),
])
np.savetxt(
    "loss_vs_iteration.csv",
    loss_curve_df,
    delimiter=",",
    header="iteration,lasso_loss,ridge_loss,standard_loss",
    comments=""
)

lambda_values = np.linspace(0.00, 4, 100)

sweep_results = lambda_sweep_accuracy(
    X,
    y_folds,
    y_zero_vs_rest,
    y_one_vs_rest,
    y_two_vs_rest,
    beta_init,
    lambda_values
)

best_lasso_idx = int(np.argmax(sweep_results["lasso"]))
best_ridge_idx = int(np.argmax(sweep_results["ridge"]))



lambda_curve_df = np.column_stack([
    sweep_results["lambda_values"],
    sweep_results["lasso"],
    sweep_results["ridge"],
    sweep_results["standard"],
])
np.savetxt(
    "lambda_accuracy_curve.csv",
    lambda_curve_df,
    delimiter=",",
    header="lambda,lasso_accuracy,ridge_accuracy,standard_accuracy",
    comments=""
)

plot_lambda_accuracy_curves(sweep_results)
