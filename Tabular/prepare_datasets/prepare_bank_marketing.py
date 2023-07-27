import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = 100

SEED = 2

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / "raw_data/bank_marketing"
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data/bank_marketing"

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

bankm = pd.read_csv(RAW_DATA_DIR / "bank-additional-full.csv", sep=";")
bankm.drop("duration", axis=1, inplace=True)

bankm["target"] = (bankm["y"].apply(lambda x: x == "yes")).astype(int)
bankm.drop("y", axis=1, inplace=True)

bankm_train, bankm_test = train_test_split(
    bankm, random_state=SEED, test_size=0.2, stratify=bankm.target
)
bankm_val, bankm_test = train_test_split(
    bankm_test, random_state=SEED, test_size=0.5, stratify=bankm_test.target
)

bankm_train.to_pickle(PROCESSED_DATA_DIR / "bankm_train.p")
bankm_val.to_pickle(PROCESSED_DATA_DIR / "bankm_val.p")
bankm_test.to_pickle(PROCESSED_DATA_DIR / "bankm_test.p")
