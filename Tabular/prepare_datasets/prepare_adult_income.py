import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = 100

SEED = 1

ROOT_DIR = Path(os.getcwd())

RAW_DATA_DIR = ROOT_DIR / "raw_data/adult"
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data/adult"

if not os.path.isdir(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

colnames = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
colnames = [c.replace("-", "_") for c in colnames]

adult_train = pd.read_csv(RAW_DATA_DIR / "adult.data", names=colnames)
adult_test = pd.read_csv(RAW_DATA_DIR / "adult.test", skiprows=1, names=colnames)
adult = pd.concat([adult_train, adult_test])
adult = adult.replace(to_replace=" ?", value=np.nan).dropna()

adult = adult.sample(frac=1).reset_index(drop=True)

for c in adult.columns:
    try:
        adult[c] = adult[c].str.lower()
    except AttributeError:
        pass

adult["target"] = (adult["income"].apply(lambda x: ">50" in x)).astype(int)
adult.drop("income", axis=1, inplace=True)

adult_train, adult_test = train_test_split(
    adult, test_size=0.2, random_state=SEED, stratify=adult.target
)
adult_val, adult_test = train_test_split(
    adult_test, test_size=0.5, random_state=SEED, stratify=adult_test.target
)

adult_train.to_pickle(PROCESSED_DATA_DIR / "adult_train.p")
adult_val.to_pickle(PROCESSED_DATA_DIR / "adult_val.p")
adult_test.to_pickle(PROCESSED_DATA_DIR / "adult_test.p")
