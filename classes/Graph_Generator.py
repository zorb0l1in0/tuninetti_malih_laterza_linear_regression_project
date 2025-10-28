import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class GraphGenerator:
    def __init__(self, df: pd.DataFrame):
        """
        df: DataFrame completo (train + test)
        Se include 'SalePrice', la usa come variabile target.
        """
        self.df = df
        self.has_target = 'SalePrice' in df.columns


    def target_distribution(self, log_transform=False):
        """Mostra la distribuzione del target (se presente)."""
        if not self.has_target:
            print("⚠️ Nessuna colonna 'SalePrice' trovata nel dataset.")
            return

        data = self.df['SalePrice']
        if log_transform:
            data = np.log1p(data)

        sns.histplot(data, kde=True, bins=30)
        plt.title("Distribuzione di SalePrice" + (" (log trasformato)" if log_transform else ""))
        plt.xlabel("SalePrice")
        plt.ylabel("Frequenza")
        print(plt.show())

    def correlation_matrix(self):
        """Mostra la heatmap di correlazione solo per le colonne numeriche."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Matrice di correlazione (solo variabili numeriche)")
        plt.show()

    def scatter_feature(self, feature):
        """Scatterplot tra una feature numerica e SalePrice."""
        if not self.has_target:
            print("⚠️ Non puoi creare scatter plot: manca 'SalePrice'.")
            return
        if feature not in self.df.columns:
            print(f"⚠️ Colonna '{feature}' non trovata.")
            return

        sns.scatterplot(x=feature, y='SalePrice', data=self.df, alpha=0.6)
        plt.title(f"{feature} vs SalePrice")
        plt.show()

    def boxplot_feature(self, feature):
        """Boxplot di una feature categorica rispetto a SalePrice."""
        if not self.has_target:
            print("⚠️ Non puoi creare boxplot: manca 'SalePrice'.")
            return
        if feature not in self.df.columns:
            print(f"⚠️ Colonna '{feature}' non trovata.")
            return

        sns.boxplot(x=feature, y='SalePrice', data=self.df)
        plt.title(f"{feature} vs SalePrice")
        plt.xticks(rotation=45)
        plt.show()

    def outlier_check(self):
        """Boxplot per controllare gli outlier di SalePrice."""
        if not self.has_target:
            print("⚠️ Nessuna colonna 'SalePrice' trovata nel dataset.")
            return

        sns.boxplot(x=self.df['SalePrice'])
        plt.title("Outlier di SalePrice")
        plt.show()



if __name__ == "__main__":
    import pandas as pd
    from types import SimpleNamespace

    # 1️⃣ Carica i dati (train e test)
    train = pd.read_csv("../house_prices_data/train.csv")
    test = pd.read_csv("../house_prices_data/test.csv")

    # 2️⃣ Crea un oggetto dataset fittizio con attributi train e test
    dataset = SimpleNamespace(train=train, test=test)

    # 3️⃣ Istanzia la classe GraphGenerator
    graphs = GraphGenerator(dataset)

    # 4️⃣ Esegui alcuni test
    print("✅ Test: distribuzione del target")
    graphs.target_distribution()

    print("✅ Test: distribuzione log-trasformata")
    graphs.target_distribution(log_transform=True)

    print("✅ Test: matrice di correlazione")
    graphs.correlation_matrix()

    print("✅ Test: scatter di GrLivArea vs SalePrice")
    graphs.scatter_feature("GrLivArea")

    print("✅ Test: boxplot di OverallQual vs SalePrice")
    graphs.boxplot_feature("OverallQual")

    print("✅ Test: outlier di SalePrice")
    graphs.outlier_check()

