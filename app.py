from classes.Kaggle_loader import KaggleLoader
from classes.Data_Preprocessor import DataPreprocessor
from classes.Graph_Generator import GraphGenerator



# STEP 1 - Caricamento dataset housing prices + stampa info descrittive/statistiche
loader = KaggleLoader("competitions/house-prices-advanced-regression-techniques", "./house_prices_data")
#loader.load()
loader.print_information()
target_distribution = GraphGenerator(loader.get_full_dataset())
target_distribution.target_distribution()


# STEP - Preparazione e pulizia dei dati