import os
import pickle
import sys
from copy import copy
from datetime import datetime
from pathlib import Path
from time import time
from typing import Union

import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset as lgbDataset
from pytorch_widedeep.utils import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

sys.path.append(
    os.path.abspath("/Users/javier/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
from general_utils.lightgbm_optimizer import (  # isort:skipimport pickle  # noqa: E402
    LGBOptimizerHyperopt,
    LGBOptimizerOptuna,
)

pd.options.display.max_columns = 100


# ROOTDIR = Path("/home/ubuntu/Projects/tabulardl-benchmark")
ROOTDIR = Path("/Users/javier/Projects/tabulardl-benchmark/")
# WORKDIR = Path(os.getcwd())
WORKDIR = Path("/Users/javier/Projects/tabulardl-benchmark/run_experiments")

PROCESSED_DATA_DIR = ROOTDIR / "processed_data/bank_marketing/"

RESULTS_DIR = WORKDIR / "results/bank_marketing/lightgbm"
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

MODELS_DIR = WORKDIR / "models/bank_marketing/lightgbm"
if not MODELS_DIR.is_dir():
    os.makedirs(MODELS_DIR)

OPTIMIZE_WITH = "optuna"

train = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_val.p")
test = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_test.p")

cat_cols = []
for col in train.columns:
    if train[col].dtype == "O" or train[col].nunique() < 400 and col != "target":
        cat_cols.append(col)

num_cols = [c for c in train.columns if c not in cat_cols + ["target"]]

#  TRAIN/VALID for hyperparam optimization
label_encoder = LabelEncoder(cat_cols)
train_le = label_encoder.fit_transform(train)
valid_le = label_encoder.transform(valid)

lgbtrain = lgbDataset(
    data=train_le[cat_cols],
    label=train_le.target,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgbvalid = lgbDataset(
    data=valid_le[cat_cols],
    label=valid_le.target,
    reference=lgbtrain,
    free_raw_data=False,
)

if OPTIMIZE_WITH == "optuna":
    optimizer: Union[LGBOptimizerHyperopt, LGBOptimizerOptuna] = LGBOptimizerOptuna()
elif OPTIMIZE_WITH == "hyperopt":
    optimizer = LGBOptimizerHyperopt(verbose=True)

optimizer.optimize(lgbtrain, lgbvalid)

# Final TRAIN/TEST

ftrain = pd.concat([train, valid]).reset_index(drop=True)
flabel_encoder = LabelEncoder(cat_cols)
ftrain_le = flabel_encoder.fit_transform(ftrain)
test_le = flabel_encoder.transform(test)

params = copy(optimizer.best)
params["n_estimators"] = 1000

flgbtrain = lgbDataset(
    ftrain_le[cat_cols],
    ftrain_le.target,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgbtest = lgbDataset(
    test_le[cat_cols],
    test_le.target,
    reference=flgbtrain,
    free_raw_data=False,
)

start = time()
model = lgb.train(
    params,
    flgbtrain,
    valid_sets=[lgbtest],
    early_stopping_rounds=50,
    verbose_eval=True,
)
runtime = time() - start

preds = (model.predict(lgbtest.data) > 0.5).astype("int")
acc = accuracy_score(lgbtest.label, preds)
auc = roc_auc_score(lgbtest.label, preds)
f1 = f1_score(lgbtest.label, preds)
print(f"Accuracy: {acc}. F1: {f1}. ROC_AUC: {auc}")
print(confusion_matrix(lgbtest.label, preds))

# SAVE
suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
results_filename = "_".join(["bankm_lightgbm", suffix]) + ".p"
results_d = {}
results_d["best_params"] = optimizer.best
results_d["runtime"] = runtime
results_d["acc"] = acc
results_d["auc"] = auc
results_d["f1"] = f1
with open(RESULTS_DIR / results_filename, "wb") as f:
    pickle.dump(results_d, f)

model_filename = "_".join(["model_bankm_lightgbm", suffix]) + ".p"
with open(MODELS_DIR / model_filename, "wb") as f:
    pickle.dump(model, f)
