import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd


class KaggleLoader:
    def __init__(self, dataset: str, download_dir: str = "./data"):
        """
        Inizializza il loader con il nome del dataset e la directory di destinazione.
        Esempio: "competitions/house-prices-advanced-regression-techniques"
        """
        self.dataset = dataset
        self.download_dir = download_dir
        self.api = KaggleApi()
        self.api.authenticate()

    def load(self) -> str:
        """
        Scarica e decomprime il dataset della competizione Kaggle.
        """
        os.makedirs(self.download_dir, exist_ok=True)
        comp_name = self.dataset.split("/")[-1]

        print(f"⬇️  Downloading competition dataset: {comp_name}")
        self.api.competition_download_files(comp_name, path=self.download_dir)

        # trova il file zip
        zip_path = os.path.join(self.download_dir, f"{comp_name}.zip")

        if os.path.exists(zip_path):
            print("📦 Estrazione file...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.download_dir)
            os.remove(zip_path)
            print(f"✅ Dataset estratto in: {os.path.abspath(self.download_dir)}")
        else:
            print("⚠️ Nessun file zip trovato. Controlla di aver accettato le regole della competizione su Kaggle.")

        return os.path.abspath(self.download_dir)

    def print_information(self):
        """
        Carica i file train.csv e test.csv dalla cartella di download,
        mostra un'anteprima e le statistiche descrittive per ciascuno.
        """
        # Percorsi assoluti dei due file attesi
        train_path = os.path.join(self.download_dir, "train.csv")
        test_path = os.path.join(self.download_dir, "test.csv")

        # Controlla l'esistenza dei file
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("❌ Non sono stati trovati entrambi i file train.csv e test.csv nella cartella.")
            print(f"📁 Cartella controllata: {os.path.abspath(self.download_dir)}")
            return None, None

        # Carica i dataset
        df_train = pd.read_csv(train_path)
        #df_test = pd.read_csv(test_path)

        # --- TRAIN ---
        print("\n=== 🔹 TRAIN.CSV ===")
        print(f"Dimensioni: {df_train.shape[0]} righe × {df_train.shape[1]} colonne")
        print("\n📄 Prime righe:")
        print(df_train.head())
        print("\n📊 Statistiche descrittive:")
        print(df_train.describe(include='all'))
        #Conteggio di tipologie di variabili
        print(f"\n\n Ci sono {df_train.select_dtypes(include=['object']).shape[1]} variabili categoriche, {df_train.select_dtypes(include=['number']).shape[1]} variabili numeriche, {df_train.select_dtypes(include=['bool']).shape[1]} variabili boleane.")
        print(f"\n\n Tipi di dati per colonna: \n{df_train.dtypes}")
        print(f"\n\n Valori nulli per colonna: \n{df_train.isnull().sum()}")


        ## --- TEST ---
        #print("\n=== 🔸 TEST.CSV ===")
        #print(f"Dimensioni: {df_test.shape[0]} righe × {df_test.shape[1]} colonne")
        #print("\n📄 Prime righe:")
        #print(df_test.head())
        #print("\n📊 Statistiche descrittive:")
        #print(df_test.describe(include='all'))

    def get_full_dataset(self):
        # Percorso assolutoi del file train.csv
        train_path = os.path.join(self.download_dir, "train.csv")
        test_path = os.path.join(self.download_dir, "test.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        print(df_train.shape)
        return df_train, df_test




if __name__ == "__main__":
    loader = KaggleLoader("competitions/house-prices-advanced-regression-techniques", "../house_prices_data")
    #loader.load()
    loader.print_information()
    loader.get_full_dataset()

