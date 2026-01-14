"""
Constants used throughout the program.
    FEATURES - features used by model
    TARGET_COL - target model column name
    REQUIRED_COLS - required columns in CSV file uploaded for fraud prediction
"""
FEATURES = ["Transaction Amount", "Account Age Days"]
TARGET_COL = "Is Fraudulent"
REQUIRED_COLS = {"id", "amount", "account_age"}
