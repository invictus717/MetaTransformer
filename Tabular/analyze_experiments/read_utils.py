import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from constants import *  # noqa: F401, F403

sys.path.append(
    os.path.abspath("/home/ubuntu/Projects/tabulardl-benchmark/run_experiments")
)  # isort:skipimport pickle
import general_utils  # noqa: E402, F401

pd.options.display.max_columns = 100

ROOT_DIR = Path("/home/ubuntu/Projects/tabulardl-benchmark/")
RESULTS_DIR = ROOT_DIR / "run_experiments/results"
BEST_MODELS_DIR = ROOT_DIR / "run_experiments/best_models"


def model_results_df(
    dataset: str, keys_to_keep: List[str], model_name: str, top_n: int = None
):

    res_l = _read_dataset_model_results(
        dataset=dataset, keys_to_keep=keys_to_keep, model_name=model_name
    )

    res_df = pd.concat([pd.DataFrame(d, index=[0]) for d in res_l], ignore_index=True)
    res_df = res_df.sort_values("val_loss").reset_index(drop=True)

    return res_df.head(top_n) if top_n is not None else res_df


def lightgbm_vs_dl_df(
    dataset: str, metric: List[str], sort_by_metric: str, ascending: bool
):

    model_res_d = _lightgbm_vs_dl(dataset, metric)

    models = list(model_res_d.keys())
    cols = ["model"] + list(model_res_d[models[0]].keys())

    vals = []
    for model in models:
        vals.append(np.array(list(model_res_d[model].values())))

    final_df = pd.concat([pd.DataFrame(models), pd.DataFrame(np.vstack(vals))], axis=1)
    final_df.columns = cols
    final_df.rename(columns={"best_epoch": "best_epoch_or_ntrees"}, inplace=True)

    return final_df.sort_values(sort_by_metric, ascending=ascending, ignore_index=True)


def _read_dataset_model_results(dataset: str, keys_to_keep: List[str], model_name: str):

    res_dir = RESULTS_DIR / dataset / model_name
    res_l = []

    for fn in (res_dir).glob("*"):
        if "best" not in str(fn):
            with open(fn, "rb") as f:
                res_d = pickle.load(f)
            res = {k: res_d["args"][k] for k in keys_to_keep}
            res["val_loss"] = res_d["early_stopping"].best
            res_l.append(res)

    return res_l


def _read_dataset_best_model_results(dataset: str, model_name: str):
    res_dir = RESULTS_DIR / dataset / model_name
    best_res_fn = sorted([fn for fn in list(res_dir.glob("*")) if "best" in str(fn)])[
        -1
    ]
    with open(best_res_fn, "rb") as f:
        best_res = pickle.load(f)
    return best_res


def _lightgbm_vs_dl(dataset: str, metric: List[str]):

    res_dirs = [
        RESULTS_DIR / dataset / "tabmlp",
        RESULTS_DIR / dataset / "tabresnet",
        RESULTS_DIR / dataset / "tabnet",
        RESULTS_DIR / dataset / "tabtransformer",
        RESULTS_DIR / dataset / "lightgbm",
    ]

    lgb_model_dir = BEST_MODELS_DIR / dataset / "lightgbm"
    lgb_model_path = list(lgb_model_dir.glob("*.p"))[0]
    with open(lgb_model_path, "rb") as f:
        lgb_model = pickle.load(f)

    model_res: OrderedDict = OrderedDict()
    for dir in res_dirs:
        model_name = dir.name
        if "tab" in model_name:
            best_res = _read_dataset_best_model_results(dataset, model_name)
            model_res[model_name] = {}
            for m in metric:
                model_res[model_name][m] = best_res[m]

            stopped_epoch = best_res["early_stopping"].stopped_epoch
            early_stop_patience = best_res["args"]["early_stop_patience"]
            best_epoch = 1 + stopped_epoch - early_stop_patience
            total_rutime = best_res["runtime"]
            runtime_per_epoch = total_rutime / stopped_epoch
            effective_runtime = best_epoch * runtime_per_epoch
            model_res[model_name]["runtime"] = effective_runtime
            model_res[model_name]["best_epoch"] = best_epoch
            # model_res[model_name]["runtime_per_epoch"] = runtime_per_epoch
        else:
            model_res["lightgbm"] = {}
            fn = list(dir.glob("*.p"))[0]
            with open(fn, "rb") as f:
                lgb_model_res = pickle.load(f)
            for m in metric:
                model_res["lightgbm"][m] = lgb_model_res[m]

            model_res["lightgbm"]["runtime"] = lgb_model_res["runtime"]
            model_res["lightgbm"]["n_trees"] = lgb_model.num_trees()

    return model_res
