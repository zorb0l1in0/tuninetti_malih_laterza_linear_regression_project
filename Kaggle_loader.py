import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

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


if __name__ == "__main__":
    loader = KaggleLoader("competitions/house-prices-advanced-regression-techniques", "./house_prices_data")
    loader.load()
