# main.py
from classes.Kaggle_loader import KaggleLoader
from classes.Data_Preprocessor import DataPreprocessor
from classes.Graph_Generator import GraphGenerator
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from classes.OLSRegressor import OLSRegressor
from classes.ModelEvaluator import ModelEvaluator
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ==== CONFIGURAZIONE ====
USE_CROSS_VALIDATION = True  # <-- Cambia a False per usare lo split semplice
# =========================

if __name__ == "__main__":
    ######### STEP 1 - Caricamento dataset + info
    loader = KaggleLoader("competitions/house-prices-advanced-regression-techniques", "./house_prices_data")
    # loader.load()  # Decommenta solo se non hai i dati

    loader.print_information()
    df_train, df_test = loader.get_full_dataset()

    # Grafici target
    graphGenComp = GraphGenerator(df_train)
    print("\n📊 Distribuzione originale del target:")
    graphGenComp.target_distribution(log_transform=False)

    print("\n📊 Distribuzione log-trasformata del target:")
    df_log = df_train.copy()
    df_log["SalePrice"] = np.log(df_log["SalePrice"])
    graph_log = GraphGenerator(df_log)
    graph_log.target_distribution(log_transform=False)

    ######### STEP 2 - Preprocessing
    pre = DataPreprocessor(target="SalePrice")
    X_train, X_test, y_train = pre.prepare(df_train, df_test)

    ######### STEP 3 - Verifica ipotesi BLUE
    pre.check_BLUE_assumptions()

    # ===========================================================
    # STEP 4 - VALUTAZIONE: SPLIT SEMPLICE O CROSS-VALIDATION
    # ===========================================================
    if USE_CROSS_VALIDATION:
        print("\n🔄 Avvio Cross-Validation a 5 fold...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {"RMSE": [], "MAE": [], "R²": []}
#
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            ols_cv = OLSRegressor()
            ols_cv.fit(X_tr, y_tr)
            y_pred_val = ols_cv.predict(X_val)
            metrics = ModelEvaluator.compute_metrics(y_val, y_pred_val)
            for key in cv_scores:
                cv_scores[key].append(metrics[key])
            print(f"  Fold {fold} → RMSE: {metrics['RMSE']:.6f}, R²: {metrics['R²']:.4f}")
#
        print("\n📊 METRICHE MEDIE (CV 5-fold, scala log):")
        for metric, values in cv_scores.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric}: {mean_val:.6f} ± {std_val:.6f}")
#
        # Modello finale su TUTTO il train
        ModelEvaluator.plot_residuals(y_val, y_pred_val)
        ModelEvaluator.plot_predictions(y_val, y_pred_val)

        final_model = OLSRegressor()
        final_model.fit(X_train, y_train)

        # Grafico sulla curva di apprendimento (usa GraphGenerator)
        print("\n📈 Curva di apprendimento...")
        graphGenComp.learning_curve_plot(X_train, y_train, cv=5)


    else:
        print("\n✂️ Utilizzo split semplice (80/20)...")
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        ols_model = OLSRegressor()
        ols_model.fit(X_train_split, y_train_split)

        y_pred_val = ols_model.predict(X_val)
        metrics = ModelEvaluator.compute_metrics(y_val, y_pred_val)
        print("\n📊 METRICHE SUL VALIDATION SET (scala log):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Plot opzionali (solo in split semplice)
        ModelEvaluator.plot_residuals(y_val, y_pred_val)
        ModelEvaluator.plot_predictions(y_val, y_pred_val)

        final_model = ols_model


    # ===========================================================
    # STEP 5 - SUBMISSION FINALE
    # ===========================================================
    log_preds_test = final_model.predict(X_test)
    saleprice_preds = np.expm1(log_preds_test)

    submission = pd.DataFrame({
        "Id": df_test["Id"],
        "SalePrice": saleprice_preds
    })
    submission.to_csv("generated_submission.csv", index=False)
    print("\n✅ File 'generated_submission.csv' generato con successo!")