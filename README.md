# Fraudulent-Transactions-SUML

Train : python app.py train --data .\data\Train.csv --out .\models\fraud_rf_bundle.pkl
Test/Evaluate: python app.py eval --model .\models\fraud_rf_bundle.pkl --data .\data\Test.csv