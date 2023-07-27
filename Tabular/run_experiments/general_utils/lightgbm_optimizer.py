import warnings
from typing import Any, Dict, Optional

import lightgbm as lgb
import pandas as pd
from hyperopt import Trials, fmin, hp, space_eval, tpe
from lightgbm import Dataset as lgbDataset
from optuna.integration.lightgbm import LightGBMTunerCV
from sklearn.metrics import log_loss, mean_squared_error

warnings.filterwarnings("ignore")


class LGBOptimizerHyperopt(object):
    def __init__(
        self,
        objective: str = "binary",
        is_unbalance: bool = False,
        verbose: bool = False,
        num_class: Optional[int] = None,
    ):

        self.objective = objective
        if objective == "multiclass" and not num_class:
            raise ValueError("num_class must be provided for multiclass problems")
        self.num_class = num_class
        self.is_unbalance = is_unbalance
        self.verbose = verbose
        self.early_stop_dict: Dict = {}

    def optimize(
        self,
        dtrain: lgbDataset,
        deval: lgbDataset,
        maxevals: int = 200,
    ):

        if self.objective == "regression":
            self.best = lgb.LGBMRegressor().get_params()
        else:
            self.best = lgb.LGBMClassifier().get_params()
        del (self.best["silent"], self.best["importance_type"])

        param_space = self.hyperparameter_space()
        objective = self.get_objective(dtrain, deval)
        objective.i = 0
        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=maxevals,
            trials=trials,
            verbose=self.verbose,
        )
        self.trials = trials
        best = space_eval(param_space, trials.argmin)
        best["n_estimators"] = int(best["n_estimators"])
        best["num_leaves"] = int(best["num_leaves"])
        best["min_child_samples"] = int(best["min_child_samples"])
        best["verbose"] = -1
        best["objective"] = self.objective
        self.best.update(best)

    def get_objective(self, dtrain: lgbDataset, deval: lgbDataset):
        def objective(params: Dict[str, Any]) -> float:

            # hyperopt casts as float
            params["n_estimators"] = int(params["n_estimators"])
            params["num_leaves"] = int(params["num_leaves"])
            params["min_child_samples"] = int(params["min_child_samples"])
            params["verbose"] = -1
            params["seed"] = 1

            params["feature_pre_filter"] = False

            params["objective"] = self.objective

            if self.objective != "regression":
                params["is_unbalance"] = self.is_unbalance

            if self.objective == "multiclass":
                params["num_class"] = self.num_class

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[deval],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            preds = model.predict(deval.data)

            if self.objective != "regression":
                score = log_loss(deval.label, preds)
            elif self.objective == "regression":
                score = mean_squared_error(deval.label, preds)

            objective.i += 1  # type: ignore

            return score

        return objective

    def hyperparameter_space(
        self, param_space: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
            "num_leaves": hp.quniform("num_leaves", 20, 200, 10),
            "min_child_samples": hp.quniform("min_child_samples", 20, 100, 20),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": hp.choice(
                "reg_alpha", [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]
            ),
            "reg_lambda": hp.choice(
                "reg_lambda", [0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]
            ),
        }
        if param_space:
            return param_space
        else:
            return space


class LGBOptimizerOptuna(object):
    def __init__(
        self,
        objective: str = "binary",
        is_unbalance: bool = False,
        verbose: bool = False,
        num_class: Optional[int] = None,
    ):

        self.objective = objective
        if objective == "multiclass" and not num_class:
            raise ValueError("num_class must be provided for multiclass problems")
        self.num_class = num_class
        self.is_unbalance = is_unbalance
        self.verbose = verbose
        self.best: Dict[str, Any] = {}  # Best hyper-parameters

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):
        # Define the base parameters
        if self.objective == "binary":
            params: Dict = {"objective": self.objective}
        elif self.objective == "multiclass":
            params: Dict = {"objective": self.objective, "metric": "multi_logloss"}
        elif self.objective == "regression":
            params: Dict = {"objective": self.objective, "metric": "rmse"}

        if self.verbose:
            params["verbosity"] = 1
        else:
            params["verbosity"] = -1

        if self.objective != "regression":
            params["is_unbalance"] = self.is_unbalance

        if self.objective == "multiclass":
            params["num_class"] = self.num_class

        # Reformat the data for LightGBM cross validation method
        train_set = lgb.Dataset(
            data=pd.concat([dtrain.data, deval.data]).reset_index(drop=True),
            label=pd.concat([dtrain.label, deval.label]).reset_index(drop=True),
            categorical_feature=dtrain.categorical_feature,
            free_raw_data=False,
        )
        train_index = range(len(dtrain.data))
        valid_index = range(len(dtrain.data), len(train_set.data))

        # Run the hyper-parameter tuning
        self.tuner = LightGBMTunerCV(
            params=params,
            train_set=train_set,
            folds=[(train_index, valid_index)],
            verbose_eval=False,
            num_boost_round=1000,
            early_stopping_rounds=50,
        )

        self.tuner.run()

        self.best = self.tuner.best_params
        # since n_estimators is not among the params that Optuna optimizes we
        # need to add it manually. We add a high value since it will be used
        # with early_stopping_rounds
        self.best["n_estimators"] = 1000  # type: ignore
