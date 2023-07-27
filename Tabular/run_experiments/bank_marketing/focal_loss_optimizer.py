import torch
from hyperopt import Trials, fmin, hp, tpe
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
from pytorch_widedeep.metrics import Accuracy, F1Score
from pytorch_widedeep.models import TabMlp, WideDeep


class FLOptimizer:
    def hyperparameter_space(self):

        space = {
            "alpha": hp.uniform("alpha", 0.1, 0.8),
            "gamma": hp.uniform("gamma", 0.5, 5.0),
        }
        return space

    def optimize(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        prepare_tab,
        mlp_hidden_dims,
        args,
        maxevals,
    ):

        param_space = self.hyperparameter_space()
        objective = self.get_objective(
            X_train, y_train, X_valid, y_valid, prepare_tab, mlp_hidden_dims, args
        )

        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=maxevals,
            trials=trials,
        )

        self.best = best

    def get_objective(
        self, X_train, y_train, X_valid, y_valid, prepare_tab, mlp_hidden_dims, args
    ):
        def objective(params):

            deeptabular = TabMlp(
                column_idx=prepare_tab.column_idx,
                mlp_hidden_dims=mlp_hidden_dims,
                mlp_activation=args.mlp_activation,
                mlp_dropout=args.mlp_dropout,
                mlp_batchnorm=args.mlp_batchnorm,
                mlp_batchnorm_last=args.mlp_batchnorm_last,
                mlp_linear_first=args.mlp_linear_first,
                embed_input=prepare_tab.embeddings_input,
                embed_dropout=args.embed_dropout,
            )
            model = WideDeep(deeptabular=deeptabular)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=args.rop_mode,
                factor=args.rop_factor,
                patience=args.rop_patience,
                threshold=args.rop_threshold,
                threshold_mode=args.rop_threshold_mode,
            )

            early_stopping = EarlyStopping(
                monitor=args.monitor,
                min_delta=args.early_stop_delta,
                patience=args.early_stop_patience,
            )
            trainer = Trainer(
                model,
                objective="binary_focal_loss",
                optimizers=optimizer,
                lr_schedulers=lr_scheduler,
                reducelronplateau_criterion=args.monitor.split("_")[-1],
                callbacks=[early_stopping, LRHistory(n_epochs=args.n_epochs)],
                metrics=[Accuracy, F1Score],
                alpha=params["alpha"],
                gamma=params["gamma"],
                verbose=0,
            )

            trainer.fit(
                X_train={"X_tab": X_train, "target": y_train},
                X_val={"X_tab": X_valid, "target": y_valid},
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                validation_freq=args.eval_every,
            )

            score = early_stopping.best

            return score

        return objective
