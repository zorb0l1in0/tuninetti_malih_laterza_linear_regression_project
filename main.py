from classes.Kaggle_loader import KaggleLoader
from classes.Data_Preprocessor import DataPreprocessor
from classes.Graph_Generator import GraphGenerator
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


if __name__ == "__main__":
    #########  STEP 1 - Caricamento dataset housing prices + stampa info descrittive/statistiche
    loader = KaggleLoader("competitions/house-prices-advanced-regression-techniques", "./house_prices_data")

    #Decommentare se non si ha il dataset in locale e serve scaricarselo
    #loader.load()

    # Stampa informazioni sul dataset
    loader.print_information()

    # Ottieni il dataset di train
    df_train, df_test = loader.get_full_dataset()

    # Istanzia la classe GraphGenerator
    graphGenComp = GraphGenerator(df_train)

    # Osservazione iniziale della distribuzione del target 'SalePrice'
    target_distribution_graph = graphGenComp.target_distribution()

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

    model = pre.check_BLUE_assumptions()
