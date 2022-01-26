import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Basic args.
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help=("Path to pretrained model or model identifier from "
              "huggingface.co/models"),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=("The output directory where the model predictions and "
              "checkpoints will be written."),
    )
    parser.add_argument(
        "--output_root",
        default="./outputs",
        type=str,
        required=False,
        help=("The output root directory."),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=False,
        help=("The names of the tasks, see data_processing/*_processors.py"),
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help=("The dir of the datasets, see data_processing/*_processors.py"),
    )
    parser.add_argument(
        "--do_not_load_optimizer",
        action="store_true",
        help=("If prohibit loading the optimizer state dict (using pretrain)."),
    )
    parser.add_argument(
        "--do_not_load_pretrained_model_weights",
        action="store_true",
        help=("If prohibit loading the optimizer state dict (using pretrain)."),
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "dev", "test"],
        default="dev",
        help=("The splits to evaluate on."),
    )
    parser.add_argument(
        "--score_average_method",
        default="binary",
        type=str,
        required=False,
        choices=["macro", "micro", "binary"],
        help=("For sklearn scores."),
    )

    # Pretraining args.
    parser.add_argument(
        "--mlm_probability",
        default=0.3,
        type=float,
        required=False,
        help=("The MLM probability."),
    )
    parser.add_argument(
        "--mlm_ignore_index",
        default=-100,
        type=int,
        required=False,
        help=("The MLM CE loss ignored index."),
    )
    parser.add_argument(
        "--num_labels",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--training_phase",
        default="finetune",
        type=str,
        required=False,
        choices=["finetune", "pretrain"],
        help=("The phase for the training."),
    )

    # Other args for training.
    parser.add_argument(
        "--config_name", default="", type=str,
        help=("Pretrained config name "
              "or path if not the same as model_name")
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help=("Pretrained tokenizer name or path "
              "if not the same as model_name"),
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=("The maximum total input sequence length after tokenization. "
              "Sequences longer than this will be truncated, sequences "
              "shorter will be padded."),
    )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true",
        help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", 
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing "
             "a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. "
             "Override num_train_epochs.",
    )
    parser.add_argument(
        "--max_eval_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of evaluating steps to perform.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as "
             "model_name ending and ending with step number",
    )
    parser.add_argument("--iters_to_eval", default=None, type=str, nargs="+",
                        help="Iterations of checkpoints to evaluate.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision "
             "(through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in "
             "['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="",
                        help="For distant debugging.")
    parser.add_argument(
        "--no_gene", action="store_true", 
        help="Set this flag if not using the gene and variation information."
    )
    args = parser.parse_args()

    return args
