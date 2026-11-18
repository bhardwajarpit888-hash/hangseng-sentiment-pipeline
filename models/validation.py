import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import binomtest


class OutOfSampleValidator:
    """
    Executes walk-forward out-of-sample cross validation without lookahead bias.
    Enforces minimum sample sizes, purged gap between train/test folds,
    naive baseline benchmarking, statistical significance tests, and confidence intervals.
    """

    def __init__(self, n_splits: int = 3, min_train: int = 40, min_test: int = 15, gap: int = 1):
        self.n_splits = n_splits
        self.min_train = min_train
        self.min_test = min_test
        self.gap = gap

    def evaluate_walk_forward(
        self,
        model_builder_fn,
        X: pd.DataFrame,
        y: pd.Series,
        prev_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Runs TimeSeriesSplit with gap to prevent overlap leakage.
        Returns None for metrics if sample size requirements are not met.
        """
        n_samples = len(X)
        if n_samples < (self.min_train + self.min_test):
            return self._insufficient_data_response(n_samples)

        # TimeSeriesSplit with gap parameter to prevent lookahead leakage
        try:
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                gap=self.gap,
                max_train_size=None,
                test_size=max(self.min_test, n_samples // (self.n_splits + 2))
            )
        except TypeError:
            # Fallback if gap argument is unsupported in older scikit-learn
            tscv = TimeSeriesSplit(n_splits=self.n_splits)

        rmses, maes = [], []
        y_preds_all, y_trues_all = [], []
        naive_prev_sign_preds = []
        valid_folds = 0

        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < self.min_train or len(test_idx) < self.min_test:
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = model_builder_fn()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))

            rmses.append(rmse)
            maes.append(mae)
            y_preds_all.extend(preds.tolist())
            y_trues_all.extend(y_test.tolist())

            # Naive baseline: persistence of previous return sign
            if prev_returns is not None and len(prev_returns) == len(X):
                naive_prev_sign_preds.extend(prev_returns.iloc[test_idx].tolist())
            else:
                # Shifted by 1 in test set
                naive_prev_sign_preds.extend([0.0] + y_test.tolist()[:-1])

            valid_folds += 1

        total_test_points = len(y_trues_all)
        if total_test_points < self.min_test or valid_folds == 0:
            return self._insufficient_data_response(total_test_points)

        y_preds_arr = np.array(y_preds_all)
        y_trues_arr = np.array(y_trues_all)

        # Model directional accuracy
        hits = (np.sign(y_preds_arr) == np.sign(y_trues_arr)).astype(int)
        k_success = int(np.sum(hits))
        hit_ratio = float(k_success / total_test_points)

        # Naive Baseline 1: Zero return prediction
        naive_zero_mae = float(mean_absolute_error(y_trues_arr, np.zeros_like(y_trues_arr)))
        naive_zero_rmse = float(np.sqrt(mean_squared_error(y_trues_arr, np.zeros_like(y_trues_arr))))

        # Naive Baseline 2: Persistence (predict same sign as previous return)
        naive_hits = (np.sign(np.array(naive_prev_sign_preds)) == np.sign(y_trues_arr)).astype(int)
        naive_persistence_accuracy = float(np.mean(naive_hits))

        # Model lift vs naive persistence baseline
        accuracy_lift = float(hit_ratio - naive_persistence_accuracy)

        # Binomial test for statistical significance (H0: hit rate = 0.50, H1: hit rate > 0.50)
        binom_res = binomtest(k_success, n=total_test_points, p=0.5, alternative="greater")
        p_value = float(binom_res.pvalue)

        # 95% Wilson confidence interval for directional accuracy
        ci = binom_res.proportion_ci(confidence_level=0.95, method="wilson")
        ci_lower = float(ci.low)
        ci_upper = float(ci.high)

        # Sharpe ratio: calculated ONLY if n_test >= 100 to avoid spurious tiny-sample estimates
        if total_test_points >= 100:
            strategy_returns = [r if p > 0 else 0.0 for p, r in zip(y_preds_all, y_trues_all)]
            strat_series = pd.Series(strategy_returns)
            std = float(strat_series.std())
            sharpe = float((strat_series.mean() / (std + 1e-8)) * np.sqrt(252 * 6.5)) if std > 0 else None
            strategy_cum_return = float(np.prod(1 + np.array(strategy_returns)) - 1.0)
            benchmark_cum_return = float(np.prod(1 + y_trues_arr) - 1.0)
        else:
            sharpe = None
            strategy_cum_return = None
            benchmark_cum_return = None

        return {
            "status": "valid",
            "n_test": total_test_points,
            "n_folds": valid_folds,
            "mean_rmse": round(float(np.mean(rmses)), 6),
            "mean_mae": round(float(np.mean(maes)), 6),
            "directional_accuracy": round(hit_ratio, 4),
            "naive_baseline_accuracy": round(naive_persistence_accuracy, 4),
            "accuracy_lift_vs_baseline": round(accuracy_lift, 4),
            "naive_zero_mae": round(naive_zero_mae, 6),
            "naive_zero_rmse": round(naive_zero_rmse, 6),
            "p_value": round(p_value, 5),
            "statistically_significant": bool(p_value < 0.05),
            "ci_95": {
                "lower": round(ci_lower, 4),
                "upper": round(ci_upper, 4)
            },
            "sharpe_ratio": round(sharpe, 2) if sharpe is not None else None,
            "strategy_return": round(strategy_cum_return, 4) if strategy_cum_return is not None else None,
            "benchmark_return": round(benchmark_cum_return, 4) if benchmark_cum_return is not None else None
        }

    def _insufficient_data_response(self, n_samples: int) -> Dict[str, Any]:
        return {
            "status": "insufficient_data",
            "n_test": n_samples,
            "n_folds": 0,
            "mean_rmse": None,
            "mean_mae": None,
            "directional_accuracy": None,
            "naive_baseline_accuracy": None,
            "accuracy_lift_vs_baseline": None,
            "naive_zero_mae": None,
            "naive_zero_rmse": None,
            "p_value": None,
            "statistically_significant": False,
            "ci_95": {"lower": None, "upper": None},
            "sharpe_ratio": None,
            "strategy_return": None,
            "benchmark_return": None
        }
