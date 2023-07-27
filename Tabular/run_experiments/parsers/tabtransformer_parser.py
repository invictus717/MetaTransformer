import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="TabMlp parameters")

    # data set
    parser.add_argument(
        "--bankm_dset",
        type=str,
        default="bank_marketing",
        help="bank_marketing or bank_marketing_kaggle",
    )

    # should we use wide and text components  (only used for airbnb)
    parser.add_argument(
        "--with_wide",
        action="store_true",
    )
    parser.add_argument(
        "--with_text",
        action="store_true",
    )

    # model parameters
    parser.add_argument(
        "--embed_dropout", type=float, default=0.0, help="embeddings dropout"
    )
    parser.add_argument(
        "--full_embed_dropout",
        action="store_true",
        help="Boolean indicating if an entire embedding (i.e. the representation for one categorical"
        " column) will be dropped in the batch",
    )
    parser.add_argument(
        "--shared_embed",
        action="store_true",
        help="Boolean indicating if part of the embeddings will be shared accross categorical cols",
    )
    parser.add_argument(
        "--add_shared_embed",
        action="store_true",
        help="Boolean indicating if the shared embeddings strategy (see implementation for details)",
    )
    parser.add_argument(
        "--frac_shared_embed",
        type=int,
        default=8,
        help="dividing factor for the shared embeddings (e.g. 32/8)",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=32,
        help="embeddings input dim",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="number of attention heads",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=6,
        help="number of transformer blocks",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Internal dropout for the TranformerEncoder and the MLP",
    )
    parser.add_argument(
        "--ff_hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension of the ``FeedForward`` Layer",
    )
    parser.add_argument(
        "--transformer_activation",
        type=str,
        default="relu",
        help="one of relu, leaky_relu, gelu ",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        type=str,
        default="None",
        help="mlp hidden dims",
    )
    parser.add_argument(
        "--mlp_activation",
        type=str,
        default="relu",
        help="one of relu, leaky_relu, gelu ",
    )
    parser.add_argument(
        "--mlp_batchnorm",
        action="store_true",
        help="if true the dense layers will be built with BatchNorm",
    )
    parser.add_argument(
        "--mlp_batchnorm_last",
        action="store_true",
        help="if true BatchNorm will be applied to the last of the dense layers",
    )
    parser.add_argument(
        "--mlp_linear_first",
        action="store_true",
        help="Boolean indicating the order of the operations in the dense",
    )

    # deeptext model parameters (only used for airbnb)
    parser.add_argument(
        "--max_vocab",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=140,
    )
    parser.add_argument(
        "--pad_first",
        action="store_true",
    )
    parser.add_argument(
        "--pad_idx",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_word_vectors",
        action="store_true",
    )
    parser.add_argument(
        "--prepare_text",
        action="store_true",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="lstm",
        help="one of lstm and gru",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="rnn hidden dim",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="rnn numb of layers",
    )
    parser.add_argument(
        "--rnn_dropout",
        type=float,
        default=0.1,
        help="rnn dropout",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="rnn bidirectional",
    )
    parser.add_argument(
        "--use_hidden_state",
        action="store_true",
        help="whether to use the last hidden state or the rnn output as predictors",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=300,
        help="Dimension of the word embedding matrix if non-pretained word vectors are used",
    )
    parser.add_argument(
        "--embed_trainable",
        action="store_true",
        help="Boolean indicating if the pretrained embeddings are trainable",
    )
    parser.add_argument(
        "--head_hidden_dims",
        type=str,
        default="None",
    )
    parser.add_argument(
        "--head_activation",
        type=str,
        default="relu",
        help="one of relu, leaky_relu, gelu ",
    )
    parser.add_argument("--head_dropout", type=float, default=0.1, help="mlp dropout")
    parser.add_argument(
        "--head_batchnorm",
        action="store_true",
        help="if true the dense layers will be built with BatchNorm",
    )
    parser.add_argument(
        "--head_batchnorm_last",
        action="store_true",
        help="if true BatchNorm will be applied to the last of the dense layers",
    )
    parser.add_argument(
        "--head_linear_first",
        action="store_true",
        help="Boolean indicating the order of the operations in the dense",
    )

    # warming up parameters (only used for airbnb)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="warmup model components before the joined learning stars",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--warmup_max_lr",
        type=float,
        default=0.01,
    )

    # train/eval parameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epoch.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 reg.")
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--early_stop_delta",
        type=float,
        default=0.0,
        help="Min delta for early stopping",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=30,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="(val_)loss or (val_)metric name to monitor",
    )

    # Optimizer parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Only Adam, AdamW, and RAdam are considered. UseDefault is AdamW with default values",
    )

    # Scheduler parameters
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="ReduceLROnPlateau",
        help="one of 'ReduceLROnPlateau', 'CyclicLR' or 'OneCycleLR', NoScheduler",
    )
    # ReduceLROnPlateau (rop) params
    parser.add_argument(
        "--rop_mode",
        type=str,
        default="min",
        help="One of min, max",
    )
    parser.add_argument(
        "--rop_factor",
        type=float,
        default=0.1,
        help="Factor by which the learning rate will be reduced",
    )
    parser.add_argument(
        "--rop_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which learning rate will be reduced",
    )
    parser.add_argument(
        "--rop_threshold",
        type=float,
        default=0.001,
        help="Threshold for measuring the new optimum",
    )
    parser.add_argument(
        "--rop_threshold_mode",
        type=str,
        default="abs",
        help="One of rel, abs",
    )
    # CyclicLR and OneCycleLR params
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.001,
        help="base_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.01,
        help="max_lr for cyclic lr_schedulers",
    )
    parser.add_argument(
        "--div_factor",
        type=float,
        default=25,
        help="Determines the initial learning rate via initial_lr = max_lr/div_factor",
    )
    parser.add_argument(
        "--final_div_factor",
        type=float,
        default=1e4,
        help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
    )
    parser.add_argument(
        "--n_cycles",
        type=float,
        default=5,
        help="number of cycles for CyclicLR",
    )
    parser.add_argument(
        "--cycle_momentum",
        action="store_true",
    )
    parser.add_argument(
        "--pct_step_up",
        type=float,
        default=0.3,
        help="Percentage of the cycle (in number of steps) spent increasing the learning rate",
    )

    # save parameters
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )

    return parser.parse_args()
