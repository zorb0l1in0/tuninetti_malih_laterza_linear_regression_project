from classes.Kaggle_loader import KaggleLoader
from classes.Data_Preprocessor import DataPreprocessor
from classes.Graph_Generator import GraphGenerator
import seaborn as sns
import pandas as pd
import numpy as np  # 👈 aggiunto per la trasformazione log

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


if __name__ == "__main__":
    #########  STEP 1 - Caricamento dataset housing prices + stampa info descrittive/statistiche
    loader = KaggleLoader("competitions/house-prices-advanced-regression-techniques", "./house_prices_data")

    # Decommentare se non si ha il dataset in locale e serve scaricarselo
    # loader.load()

    # Stampa informazioni sul dataset
    loader.print_information()

    # Ottieni il dataset di train
    df_train, df_test = loader.get_full_dataset()

    # Istanzia la classe GraphGenerator
    graphGenComp = GraphGenerator(df_train)

    # 1️⃣ Osservazione iniziale della distribuzione del target 'SalePrice'
    print("\n📊 Distribuzione originale del target:")
    graphGenComp.target_distribution(log_transform=False)

    # 2️⃣ Visualizzazione della distribuzione log-trasformata del target
    print("\n📊 Distribuzione log-trasformata del target:")
    df_log = df_train.copy()
    df_log["SalePrice"] = np.log(df_log["SalePrice"])
    graph_log = GraphGenerator(df_log)
    graph_log.target_distribution(log_transform=False)

    # NOTA IMPORTANTE: bisognerebbe osservare anche la distribuzione delle variabili numeriche continue in relazione
    # al target + eventuali altri grafici esplorativi

    ######### STEP 2 - Preparazione e pulizia dei dati
    # Rilevamento e imputazione valori nulli (mediana per numeriche, moda per categoriche)
    # Trasformazione logaritmica di SalePrice (per ridurre asimmetria e migliorare normalità dei residui)
    # Identificazione ed eventuale rimozione/winsorizzazione outlier (OLS è sensibile agli outlier)
    # Encoding variabili categoriche (one-hot encoding)
    # Controllo multicollinearità tra feature (heatmap o VIF > 5 → rimozione o aggregazione variabili)
    # Selezione feature più correlate con il target (riduzione dimensione, evita overfitting)
    # Scaling features numeriche (StandardScaler per omoschedasticità e confronto tra coefficienti)
    # Calcolo correlazioni tra variabili numeriche e target (analisi preliminare, scelta predittori rilevanti)
    # Split in train/test set (per valutazione indipendente del modello)

    pre = DataPreprocessor(target="SalePrice")

    X_train, X_test, y_train = pre.prepare(df_train, df_test)

    ######### STEP 3 - Verifica ipotesi BLUE
    pre.check_BLUE_assumptions()

    # ===========================================================
    # STEP 3 - ADDESTRAMENTO MODELLO E VALUTAZIONE (aggiunto in fondo)
    # ===========================================================
    from sklearn.model_selection import train_test_split
    from classes.OLSRegressor import OLSRegressor
    from classes.ModelEvaluator import ModelEvaluator
    import numpy as np

    # Split del train in train + validation (per valutazione)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Addestra il modello OLS
    ols_model = OLSRegressor()
    ols_model.fit(X_train_split, y_train_split)

    # Valutazione sul validation set
    y_pred_val = ols_model.predict(X_val)
    metrics = ModelEvaluator.compute_metrics(y_val, y_pred_val)
    print("\n📊 METRICHE SUL VALIDATION SET (scala log):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")

    # (Opzionale) Plot residui e predizioni
    # ModelEvaluator.plot_residuals(y_val, y_pred_val)
    # ModelEvaluator.plot_predictions(y_val, y_pred_val)

    # Predizione su test set (per submission)
    log_preds_test = ols_model.predict(X_test)
    saleprice_preds = np.expm1(log_preds_test)  # Inversa del log → scala originale

    # Crea submission
    submission = pd.DataFrame({
        "Id": df_test["Id"],
        "SalePrice": saleprice_preds
    })
    submission.to_csv("submission.csv", index=False)
    print("\n✅ File 'submission.csv' generato con successo!")