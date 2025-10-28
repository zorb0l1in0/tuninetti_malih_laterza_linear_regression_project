# ===========================================================
# 2. DATA PREPROCESSOR
# ===========================================================
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from scipy.stats import shapiro
import statsmodels.api as sm

class DataPreprocessor:
    """
    Prepara e pulisce i dati per la regressione OLS.
    Applica le stesse trasformazioni a train e test (senza fit esplicito).
    """

    def __init__(self, target: str):
        self.target = target
        self.scaler = StandardScaler()

    def _delete_columns_with_many_missing(self, df, threshold=0.25):
        '''
        Rimuove le colonne con una percentuale di valori mancanti superiore alla soglia specificata.
        '''
        missing_fraction = df.isnull().mean()
        cols_to_drop = missing_fraction[missing_fraction > threshold].index
        print(f"Colonne rimosse per troppi valori mancanti (> {threshold*100}%): {cols_to_drop.tolist()}")
        return df.drop(columns=cols_to_drop)

    # ===============================================================
    # STEP 1 - Imputazione valori mancanti
    # ===============================================================
    def _impute_missing(self, df):
        '''
        Per le colonne numeriche imputiamo con la mediana, per le colonne categoriche con la moda.
        '''
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

    # ===============================================================
    # STEP 2 - Trasformazione logaritmica del target (solo train)
    # ===============================================================
    def _transform_target(self, df):
        '''
        Applichiamo la trasformazione logaritmica al target per ridurre asimmetria e migliorare normalità dei residui.
        '''
        if self.target in df.columns:
            if (df[self.target] <= 0).any():
                df[self.target] = np.log1p(df[self.target])
            else:
                df[self.target] = np.log(df[self.target])
        return df

    # ===============================================================
    # STEP 3 - Winsorizzazione outlier
    # ===============================================================
    def _winsorize_outliers(self, df):
        '''
        Applichiamo la tecnica di winsorizzazione al 90 percentile per limitare l'influenza degli outlier sulle variabili numeriche.
        '''
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col != self.target:
                df[col] = winsorize(df[col], limits=[0, 0.1])
        return df

    # ===============================================================
    # STEP 4 - Encoding categoriche + bool
    # ===============================================================
    def _encode_categoricals(self, df):
        '''
        Applichiamo il one-hot encoding alle variabili categoriche e convertiamo i booleani in interi 0 e 1
        '''
        print("\n🔄 Encoding variabili categoriche e booleani...")
        df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

        bool_cols = df_encoded.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

        df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
        return df_encoded

    # ===============================================================
    # STEP 5 - Rimozione feature col VIF alto
    # ===============================================================
    def _remove_high_vif(self, df):
        '''
        Calcoliamo la multicollinearità tra le variabili esplicative X usando il VIF. Questo indicatore misura quanto una
        variabile esplicativa può essere predetta dalle altre variabili esplicative. Un VIF alto (tipicamente > 5)
        indica che la variabile è altamente collineare con le altre e può essere rimossa per migliorare la stabilità del modello.
        '''
        X = df.drop(columns=[self.target], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        vif_df = pd.DataFrame()
        vif_df["feature"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        high_vif = vif_df[vif_df["VIF"] > 5]["feature"].tolist()
        print(f"Feature rimosse per VIF alto: {high_vif}")
        return df.drop(columns=high_vif, errors="ignore")

    # ===============================================================
    # STEP 6 - Selezione top feature correlate col target
    # ===============================================================
    def _select_top_features(self, df, corr_threshold=0.6):
        '''
        Seleziona le feature con correlazione (assoluta) superiore alla soglia specificata
        rispetto al target. La correlazione è calcolata con Pearson.

        Esempio:
            corr_threshold=0.6 → tiene solo le feature con r > 0.6 o r < -0.6
        '''
        # 🔹 1️⃣ Controllo che la colonna target sia nel dataframe
        if self.target not in df.columns:
            print("⚠️ Target non trovato, salto la selezione delle feature.")
            return df

        # 🔹 2️⃣ Calcola la correlazione di Pearson con il target (solo per colonne numeriche)
        corr = df.corr(numeric_only=True)[self.target].sort_values(ascending=False)

        # 🔹 3️⃣ Ordina le feature per forza di correlazione assoluta (più alta → più forte)
        corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)

        # 🔹 4️⃣ Filtra le feature con correlazione superiore alla soglia
        selected_features = corr_sorted[abs(corr_sorted) > corr_threshold].index.tolist()

        # 🔹 5️⃣ Stampa riepilogo ordinato con valori di correlazione
        print(f"\n📊 Lista delle feature ordinate per correlazione con '{self.target}':")
        print("--------------------------------------------------------")
        for feat, val in corr_sorted.items():
            mark = "✅" if abs(val) > corr_threshold else "–"
            print(f"{mark} {feat:<25} →  r = {val:>6.3f}")
        print("--------------------------------------------------------")

        # 🔹 6️⃣ Mostra solo le feature che superano la soglia
        if selected_features:
            print(f"\n✅ Feature selezionate (|r| > {corr_threshold}): {selected_features}")
        else:
            print(f"\n⚠️ Nessuna feature supera la soglia di correlazione {corr_threshold}.")

        # 🔹 7️⃣ Ritorna solo le colonne selezionate
        return df[selected_features]


    # ===============================================================
    # STEP 7 - Scaling
    # ===============================================================
    def _scale_features(self, df):
        '''
        Applichiamo lo scaling standard alle feature numeriche per garantire che abbiano media 0 e deviazione standard 1.
        '''
        if self.target in df.columns:
            X = df.drop(columns=[self.target])
        else:
            X = df
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    # ===============================================================
    # STEP 8 - BLUE CHECK (solo train)
    # ===============================================================
    def check_BLUE_assumptions(self):
        '''

        '''
        X = sm.add_constant(self.X_train)
        model = sm.OLS(self.y_train, X).fit()
        resid = model.resid

        print("\n=== Verifica ipotesi BLUE ===")
        print(f"Linearità: {'OK' if model.rsquared > 0.3 else f'KO: la metrica è {model.rsquared}, significa bassa varianza spiegata'}")
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        a = lb['lb_pvalue'].iloc[0]
        print(f"Indipendenza: {'OK' if a > 0.05 else f'KO: la metrica è {a}, significa che i residui sono correlati'}")
        bp = het_breuschpagan(resid, model.model.exog)
        print(f"Omoschedasticità: {'OK' if bp[1] > 0.05 else f'KO: la metrica è {bp[1]}, significa eteroschedasticità dei residui'}")
        sw = shapiro(resid)
        print(f"Normalità residui: {'OK' if sw.pvalue > 0.05 else f'KO : la metrica è {sw.pvalue}, significa che i residui non sono normali'}")
        print("Multicollinearità: OK (gestita via VIF)")
        return model

    # ===============================================================
    # PIPELINE PRINCIPALE — ora accetta train e test
    # ===============================================================
    def prepare(self, df_train, df_test):
        df_train = df_train.copy()
        df_test = df_test.copy()

        print("\n=== DEBUG: forme iniziali ===")
        print("Train iniziale:", df_train.shape)
        print("Test iniziale:", df_test.shape)

        # STEP 0 - Rimozione colonne con troppi missing
        df_train = self._delete_columns_with_many_missing(df_train, threshold=0.25)
        df_test = self._delete_columns_with_many_missing(df_test, threshold=0.25)

        # STEP 1 - Imputazione valori mancanti
        df_train = self._impute_missing(df_train)
        df_test = self._impute_missing(df_test)

        print("\n➡️ Dopo imputazione:")
        print(df_train.shape)
        print(df_train.columns[:10])  # solo le prime 10 per leggibilità

        # STEP 2 - Trasformazione log del target
        df_train = self._transform_target(df_train)

        # STEP 3 - Winsorizzazione outlier
        df_train = self._winsorize_outliers(df_train)
        df_test = self._winsorize_outliers(df_test)

        print("\n➡️ Dopo winsorizzazione:")
        print(df_train.shape)

        # STEP 4 - Encoding
        df_train = self._encode_categoricals(df_train)
        df_test = self._encode_categoricals(df_test)

        print("\n➡️ Dopo encoding:")
        print(df_train.shape)
        print(df_train.columns[:10])

        # STEP 5 - Rimozione VIF alto
        df_train = self._remove_high_vif(df_train)

        print("\n➡️ Dopo VIF:")
        print(df_train.shape)
        print(df_train.columns[:10])

        # STEP 6 - Selezione top feature
        df_train = self._select_top_features(df_train, corr_threshold=0.6)

        print("\n➡️ Dopo selezione feature:")
        print(df_train.shape)
        print(df_train.columns)

        # STEP 7 - Scaling
        X_train = df_train.drop(columns=[self.target], errors="ignore")
        y_train = df_train[self.target]

        if X_train.shape[1] == 0:
            raise ValueError("❌ Nessuna feature numerica rimasta dopo la selezione!")

        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)

        print("\n✅ Scaling completato:")
        print(X_train_scaled.shape)

        # Test: applica le stesse trasformazioni sulle colonne comuni
        common_cols = X_train_scaled.columns.intersection(df_test.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(df_test[common_cols]), columns=common_cols)

        print("\n✅ Fine preprocessing")
        print("Train finale:", X_train_scaled.shape)
        print("Test finale:", X_test_scaled.shape)

        return X_train_scaled, X_test_scaled, y_train

