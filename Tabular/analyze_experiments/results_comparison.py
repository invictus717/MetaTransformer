import itertools
import os
import sys
from pathlib import Path

import pandas as pd
from constants import *  # noqa: F403
from read_utils import lightgbm_vs_dl_df, model_results_df
from tqdm import tqdm

sys.path.append(os.path.abspath("../run_experiments/bank_marketing/"))

pd.options.display.max_columns = 100


LEADERBOARDS_DIR = Path("leaderboards")
LEADERBOARDS_DIR.mkdir(parents=True, exist_ok=True)


datasets = ["adult", "bank_marketing", "nyc_taxi", "fb_comments"]
models = ["tabmlp", "tabresnet", "tabnet", "tabtransformer"]
result_setup = []
for dataset, model in itertools.product(datasets, models):
    if model == "tabmlp":
        result_setup.append((dataset, model, tabmlp_keep_keys))
    if model == "tabresnet":
        result_setup.append((dataset, model, tabresnet_keep_keys))
    if model == "tabnet":
        result_setup.append((dataset, model, tabnet_keep_keys))
    if model == "tabtransformer":
        result_setup.append((dataset, model, tabtransformer_keep_keys))


for dataset, model, keys_to_keep in tqdm(result_setup):

    table_name = "_".join([dataset, model]) + ".csv"
    df = model_results_df(dataset=dataset, keys_to_keep=keys_to_keep, model_name=model)
    df.to_csv(LEADERBOARDS_DIR / table_name, index=False)


datasets_and_metrics = [
    ("adult", ["acc"], False),
    ("bank_marketing", ["f1", "auc"], False),
    ("nyc_taxi", ["rmse", "r2"], True),
    ("fb_comments", ["rmse", "r2"], True),
]
for dataset, metric, ascending in datasets_and_metrics:
    table_name = "_".join(["lightgbm_vs_dl", dataset]) + ".csv"
    df = lightgbm_vs_dl_df(
        dataset=dataset, metric=metric, sort_by_metric=metric[0], ascending=ascending
    )
    df.to_csv(LEADERBOARDS_DIR / table_name, index=False)
