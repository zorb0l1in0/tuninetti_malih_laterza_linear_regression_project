

# 🏠 House Prices: Linear Regression Project  
**Advanced Regression Techniques for Kaggle Competition**

Questo progetto implementa una **baseline robusta basata su regressione lineare OLS** per la competizione [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) su Kaggle.  
L'obiettivo è prevedere il prezzo di vendita delle case utilizzando un approccio statisticamente fondato, con verifica delle assunzioni del modello e preprocessing avanzato.

---

## 📌 Caratteristiche principali

- ✅ **Regressione lineare OLS** con `statsmodels` (non solo predizione, ma diagnosi statistica completa)
- ✅ **Verifica delle assunzioni BLUE** (linearità, omoschedasticità, normalità, assenza di multicollinearità)
- ✅ **Preprocessing avanzato**:
  - Imputazione intelligente di valori mancanti
  - Trasformazione logaritmica del target
  - Winsorizzazione degli outlier
  - Encoding di variabili categoriche
  - Rimozione feature con VIF > 5
  - Selezione feature per correlazione con il target
  - Scaling con `StandardScaler`
- ✅ **Valutazione robusta**:
  - Cross-validation a 5 fold **o** split semplice (configurabile)
  - Metriche: RMSE, MAE, R²
- ✅ **Diagnosi visiva completa**:
  - Residui vs Predizioni
  - Distribuzione dei residui
  - Distribuzione SalesPrice pre/post trasformazione logaritmica
  - Curva di apprendimento 
- ✅ **File di submission pronto per Kaggle**

---

## 📂 Struttura del progetto

```
house_prices/
├── house_prices_data/              # Dati scaricati da Kaggle
│   ├── train.csv
│   ├── test.csv
│   └── ...
├── classes/
│   ├── __init__.py
│   ├── Kaggle_loader.py            # Download e caricamento dataset
│   ├── Data_Preprocessor.py        # Pipeline completa di preprocessing
│   ├── Graph_Generator.py          # Grafici esplorativi e diagnostici
│   ├── OLSRegressor.py             # Modello OLS con statsmodels
│   └── ModelEvaluator.py           # Metriche e grafici di valutazione
├── main.py                         # Script principale (eseguibile)
├── README.md                       # Documentazione
├── requirements.txt                # Dipendenze Python
└── generated_submission.csv        # Output: submission per Kaggle
```
---

## ⚙️ Installazione

1. **Clona il repository**
   ```bash
   git clone https://github.com/zorb0l1in0/tuninetti_malih_laterza_linear_regression_project.git
   cd tuninetti_malih_laterza_linear_regression_project

2. **Crea un ambiente virtuale (opzionale ma consigliato)**
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate

3. **Installa le dipendenze**

    pip install -r requirements.txt

4. **Configura l'API di Kaggle**

    Scarica kaggle.json da Kaggle Account Settings
    Sposta il file in:
    Windows: C:\Users\<Utente>\.kaggle\kaggle.json
    Linux/macOS: ~/.kaggle/kaggle.json

    chmod 600 ~/.kaggle/kaggle.json


▶️ Esecuzione


1. **Scarica i dati (solo la prima volta)**
    Apri main.py
    Decommenta la riga: # loader.load()
    Esegui:

        python main.py

    I dati verranno salvati in ./house_prices_data/


2. **Esegui il flusso completo**

    Ricommenta loader.load()
    Scegli la modalità di valutazione in main.py

        USE_CROSS_VALIDATION = True   # True = Cross-validation, False = Split semplice

    
    Esegui:

        python main.py


3. **Risultati**

    Metriche di valutazione stampate in console
    Grafici diagnostici visualizzati automaticamente
    File generated_submission.csv creato nella root


4. **Invia su Kaggle**
    Carica generated_submission.csv su Kaggle Submission


📊 Diagnosi del modello
Il progetto include grafici avanzati per verificare le assunzioni OLS:


Residui vs Predizioni
Distribuzione dei residui
Distribuzione SalesPrice pre/post trasformazione logaritmica
Curva di Apprendimento

(Tutti i grafici sono generati automaticamente durante l'esecuzione)


📈 Metriche di valutazione

RMSE (Root Mean Squared Error): penalizza errori grandi
MAE (Mean Absolute Error): errore medio assoluto
R² (Coefficiente di determinazione): varianza spiegata

⚠️ Le metriche sono calcolate in scala logaritmica (per coerenza con il training).
La submission viene convertita automaticamente in scala originale con np.expm1(). 


🤝 Collaborazione
Il progetto è configurato per il lavoro di gruppo su Git
Usa il branch dev per lo sviluppo
Prima di pushare, esegui sempre:

        git pull origin dev



🙌 Autori

        Tuninetti Francesca
        Laterza Lorenzo
