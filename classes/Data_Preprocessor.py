# ===========================================================
# 2. DATA PREPROCESSOR CON IMPUTAZIONE
# ===========================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Import per l'imputazione
import pandas as pd


class DataPreprocessor:
    """Pulizia, encoding, imputazione e scaling dei dati."""

    def __init__(self):
        # Inizializza gli imputer per colonne numeriche e categoriche
        self.num_imputer = SimpleImputer(strategy='median')  # Per numeri usa la mediana
        self.cat_imputer = SimpleImputer(strategy='most_frequent')  # Per categorie usa il valore più frequente
        self.scaler = StandardScaler()

    def prepare(self, df, target):
        """Prepara i dati: separa target, imputa valori mancanti e fa encoding."""

        # 1. SEPARA FEATURES E TARGET
        X = df.drop(columns=[target])
        y = df[target]

        # 2. IDENTIFICA TIPI DI COLONNE
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        # 3. IMPUTAZIONE VALORI MANCANTI
        # Per colonne numeriche: riempie con la mediana
        if len(numeric_cols) > 0:
            X[numeric_cols] = self.num_imputer.fit_transform(X[numeric_cols])

        # Per colonne categoriche: riempie con il valore più frequente
        if len(categorical_cols) > 0:
            X[categorical_cols] = self.cat_imputer.fit_transform(X[categorical_cols])

        # 4. ENCODING VARIABILI CATEGORICHE
        # Converte variabili categoriche in dummy variables (one-hot encoding)
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        return X, y

    def split_and_scale(self, X, y):
        """Divide in train/test e applica scaling."""

        # 1. DIVISIONE TRAIN/TEST
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. SCALING FEATURES NUMERICHE
        # Standardizza i dati (media=0, deviazione standard=1)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
