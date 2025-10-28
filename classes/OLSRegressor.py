# model.py
import pandas as pd
import numpy as np
import statsmodels.api as sm

class OLSRegressor:
    """
    Modello di regressione lineare OLS.
    Progettato per lavorare con dati già preprocessati (log(target), scaling, encoding).
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Addestra il modello OLS.
        X: feature preprocessate (senza costante)
        y: target già log-trasformato
        """
        X = X.copy()
        X_with_const = sm.add_constant(X)
        print("Addestramento modello OLS")
        self.model = sm.OLS(y, X_with_const).fit()
        self.is_fitted = True
        self.feature_names_ = X.columns.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice i log-prezzi.
        Ritorna un array numpy di log(SalePrice).
        """
        if not self.is_fitted:
            raise ValueError("Modello non addestrato. Chiama .fit() prima.")

        X = X.copy()
        X_with_const = sm.add_constant(X)

        # Allinea le colonne con quelle usate in training (evita errori se test ha colonne mancanti/extra)
        required_cols = ['const'] + self.feature_names_
        for col in required_cols:
            if col not in X_with_const.columns:
                X_with_const[col] = 0  # imputa 0 per feature assenti (es. da one-hot)
        X_aligned = X_with_const[required_cols]

        print("Predizione logaritmo dei prezzi con modello OLS")
        return self.model.predict(X_aligned).values

    def get_summary(self):
        """Restituisce il riepilogo statistico del modello."""
        if self.model:
            return self.model.summary()
        else:
            raise ValueError("Modello non addestrato.")