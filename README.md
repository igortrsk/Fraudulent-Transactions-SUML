# Fraudulent-Transactions-SUML
Projekt **fraud detection** dla transakcji e-commerce.\
Model: **RandomForestClassifier**. Trening/predykcja opiera się na dwóch cechach:
- `Transaction Amount`
- `Account Age Days`

Etykieta:
- `Is Fraudulent` (0/1)

**Dataset:** https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions?brid=jTncmGS1ksn9sGQ3XLXEEA

## Struktura projektu
```
├──config.py                # nazwy cech i kolumny target
├──requirements.txt         # zależności
├──models
│   └──fraud_rf_bundle.pkl  # wytrenowany model na dataset z kaggle
├──apps
│   └──app.py               # CLI: train / eval / predict
│   └──streamlit_app.py     # UI w Streamlit
├──src
    └──inference.py         # Predykcja i ewaluacja
    └──preprocessing.py     # Przygotowanie i skalowanie danych
    └──training.py          # Trening RandomForest + Model Bundle
```


## Instalacja
Wymagany Python + pip.

```pip install -r requirements.txt```

## Dane
Do treningu i ewaluacji projekt potrzebuje CSV z kolumnami:
- `Transaction Amount` (float)
- `Account Age Days` (int)
- `Is Fraudulent` (0/1)

## Uruchomienie - CLI
#### Kod uruchamiany w katalogu domowym projektu
### Trening
`--data` = ścieżka do pliku CSV (musi zawierać kolumny: `Transaction Amount`, `Account Age Days`, `Is Fraudulent`)
```
python -m apps.app train --data ./sciezka/do/pliku.csv --out ./models/fraud_rf_bundle.pkl
```
### Ewaluacja
`--data` = ścieżka do pliku CSV (musi zawierać kolumny: `Transaction Amount`, `Account Age Days`, `Is Fraudulent`)
```
python -m apps.app eval --model ./models/fraud_rf_bundle.pkl --data ./sciezka/do/pliku.csv
```
### Predykcja
`--data` = ścieżka do pliku CSV, bez etykiety (kolumny: `Transaction Amount`, `Account Age Days`).\
`--out` = ścieżka do miejsca zapisu pliku wynikowego CSV.


```
python -m apps.app predict --model ./models/fraud_rf_bundle.pkl --data ./sciezka/do/pliku.csv --out ./sciezka/do/scored.csv
```

## Uruchomienie - Streamlit
Aplikacja `apps/streamlit_app.py` pozwala:
- wpisać ręcznie `Amount` i `Account Age (days)`, albo
- wgrać plik CSV z kolumnami: `id`, `amount`, `account_age`

### Start:
#### Kod uruchamiany w katalogu domowym projektu
```bash
streamlit run apps/streamlit_app.py
```

## Przykładowa tabelka CSV wgrywana do predykcji Streamlit
| id | amount | account_age |
|----|--------|-------------|
| 1  | 300,25 | 3           |
| 2  | 30     | 30          |
| 3  | 2136   | 2138        |

