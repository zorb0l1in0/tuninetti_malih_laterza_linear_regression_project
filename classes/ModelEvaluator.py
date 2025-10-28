# evaluator.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class ModelEvaluator:
    """
    Classe per valutare le prestazioni del modello.
    Supporta metriche, analisi residui e confronto predizioni vs reali.
    """

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calcola metriche di regressione.
        Nota: y_true e y_pred devono essere nella STESSA scala (es. entrambi log).
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R²": r2}

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, figsize=(10, 4)):
        """Plotta residui vs predizioni e distribuzione dei residui."""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residui vs Predizioni
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_xlabel("Predizioni (log)")
        axes[0].set_ylabel("Residui")
        axes[0].set_title("Residui vs Predizioni")

        # Distribuzione residui
        sns.histplot(residuals, kde=True, ax=axes[1])
        axes[1].set_title("Distribuzione dei residui")
        axes[1].set_xlabel("Residui")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, figsize=(6, 6)):
        """Plotta predizioni vs valori reali."""
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel("Valori reali (log)")
        plt.ylabel("Predizioni (log)")
        plt.title("Predizioni vs Reali")
        plt.show()