# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning models (e.g. Bert, DistilBERT, XLM) on specific tasks.
    Adapted from `examples/text-classification/run_xnli.py`"""

import csv
import math
import argparse
import glob
import logging
import os
import random
import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertForMaskedLM,
)

from .args import get_args
from data_processing import data_processors, data_classes
from .train_utils import mask_tokens
from .train_utils import pairwise_accuracy

# Tensorboard utilities.
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Accuracy metrics.
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Loggers.
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        output_str = args.output_dir.split("/")[-1]
        comment_str = "_{}_{}".format(output_str, args.task_name)
        tb_writer = SummaryWriter(comment=comment_str)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 \
        else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) \
                                // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")
        and not args.do_not_load_optimizer
    )):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(
                                  args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(
                                  args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github."
                              "com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed "
        "& accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if (os.path.exists(args.model_name_or_path)
        and not args.do_not_load_optimizer):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(
                args.model_name_or_path.split("/")[-1].split("-")[-1])
        except:
            global_step = 0  # If start fresh.
        epochs_trained = global_step // (len(train_dataloader) \
                         // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) \
                                         // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip"
                    " to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch",
        disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)  # Added here for reproductibility.

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            # Processes a batch.
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "labels": batch[3]}

            # Clears token type ids if using MLM-based models.
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "roberta"] else None
                )  # XLM and DistilBERT don't use segment_ids
                inputs["token_type_ids"] = None

            ##################################################
            # TODO: Please finish the following training loop.
            # Make sure to make a special if-statement for
            # args.training_phase is `pretrain`.
            raise NotImplementedError("Please finish the TODO!")

            if args.training_phase == "pretrain":
                # TODO: Mask the input tokens.
                raise NotImplementedError("Please finish the TODO!")

            # TODO: See the HuggingFace transformers doc to properly get
            # the loss from the model outputs.
            raise NotImplementedError("Please finish the TODO!")

            if args.n_gpu > 1:
                # Applies mean() to average on multi-gpu parallel training.
                loss = loss.mean()

            # Handles the `gradient_accumulation_steps`, i.e., every such
            # steps we update the model, so the loss needs to be devided.
            raise NotImplementedError("Please finish the TODO!")

            # Loss backward.
            raise NotImplementedError("Please finish the TODO!")

            # End of TODO.
            ##################################################

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.local_rank in [-1, 0] and args.logging_steps > 0
                    and global_step % args.logging_steps == 0):
                    # Log metrics
                    if (
                        # Only evaluate when single GPU otherwise metrics may
                        # not average well
                        args.local_rank == -1 and args.evaluate_during_training
                    ):
                        results = evaluate(args, model, tokenizer,
                                           data_split=args.eval_split)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_on_{}_{}".format(args.eval_split, key),
                                value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0],
                                         global_step)
                    tb_writer.add_scalar("loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step)
                    logging_loss = tr_loss

                if (args.local_rank in [-1, 0] and args.save_steps > 0
                    and global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir,
                        "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir,
                               "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s",
                                output_dir)

                    # Optional TODO: You can implement a save best functionality
                    # to also save best thus far models to a specific output 
                    # directory such as `checkpoint-best`, the saved weights
                    # will be overwritten each time your model reaches a best
                    # thus far evaluation results on the dev set.

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", data_split="test"):

    # Main evaluation loop.
    results = {}
    eval_dataset = load_and_cache_examples(args, args.task_name,
                                           tokenizer, evaluate=True,
                                           data_split=data_split,
                                           data_dir=args.data_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation on split: {} {} *****".format(
        data_split, prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    labels = None
    has_label = False

    guids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)

        if not args.do_train or (args.do_train and args.eval_split != "test"):
            guid = batch[-1].cpu().numpy()[0]
            guids.append(guid)

        with torch.no_grad():
            # Processes a batch.
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if (args.do_train and len(batch) > 3) or (not args.do_train and len(batch) > 4):
                has_label = True
                inputs["labels"] = batch[3]

            # Clears token type ids if using MLM-based models.
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "roberta"] \
                             else None
                )  # XLM and DistilBERT don't use segment_ids
                inputs["token_type_ids"] = None

            ##################################################
            # TODO: Please finish the following eval loop.
            # Make sure to make a special if-statement for
            # args.training_phase is `pretrain`.
            raise NotImplementedError("Please finish the TODO!")

            if args.training_phase == "pretrain":
                # TODO: Mask the input tokens.
                raise NotImplementedError("Please finish the TODO!")

            # TODO: See the HuggingFace transformers doc to properly get the loss
            # AND the logits from the model outputs, it can simply be 
            # indexing properly the outputs as tuples.
            # Make sure to perform a `.mean()` on the eval loss and add it
            # to the `eval_loss` variable.
            raise NotImplementedError("Please finish the TODO!")

            # TODO: Handles the logits with Softmax properly.
            raise NotImplementedError("Please finish the TODO!")

            # End of TODO.
            ##################################################

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            if has_label:
                labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if has_label:
                labels = np.append(labels,
                    inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.max_eval_steps > 0 and nb_eval_steps >= args.max_eval_steps:
            logging.info("Early stopping"
                " evaluation at step: {}".format(args.max_eval_steps))
            break

    # Organize the predictions.
    preds = np.reshape(preds, (-1, preds.shape[-1]))
    preds = np.argmax(preds, axis=-1)

    if has_label or args.training_phase == "pretrain":
        # Computes overall average eavl loss.
        eval_loss = eval_loss / nb_eval_steps

        eval_loss_dict = {"{}_loss".format(args.task_name): eval_loss}
        results.update(eval_loss_dict)

        eval_perplexity = 0
        eval_acc = 0
        eval_prec = 0
        eval_recall = 0
        eval_f1 = 0
        eval_pairwise_acc = 0

        ##################################################
        # TODO: Please finish the results computation.

        # TODO: For `pretrain` phase, we only need to compute the
        # metric "perplexity", that is the exp of the eval_loss.
        if args.training_phase == "pretrain":
            raise NotImplementedError("Please finish the TODO!")
        # TODO: Please use the preds and labels to properly compute all
        # the following metrics: accuracy, precision, recall and F1-score.
        # Please also make your sci-kit learn scores able to take the
        # `args.score_average_method` for the `average` argument.
        else:
            raise NotImplementedError("Please finish the TODO!")
            # TODO: Pairwise accuracy.
            if args.task_name == "com2sense":
                raise NotImplementedError("Please finish the TODO!")

        # End of TODO.
        ##################################################

        if args.training_phase == "pretrain":
            eval_acc_dict = {"{}_perplexity".format(args.task_name): eval_perplexity}
        else:
            eval_acc_dict = {"{}_accuracy".format(args.task_name): eval_acc}
            eval_acc_dict["{}_precision".format(args.task_name)] = eval_prec
            eval_acc_dict["{}_recall".format(args.task_name)] = eval_recall
            eval_acc_dict["{}_F1_score".format(args.task_name)] = eval_f1
            # Pairwise accuracy.
            if args.task_name == "com2sense":
                eval_acc_dict["{}_pairwise_accuracy".format(args.task_name)] = eval_pairwise_acc

        results.update(eval_acc_dict)

        output_eval_file = os.path.join(args.output_dir,
            prefix, "eval_results_split_{}.txt".format(data_split))

    if has_label:
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} on split: {} *****".format(prefix, data_split))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    # Stores the prediction .txt file to the `args.output_dir`.
    if not has_label:
        pred_file = os.path.join(args.output_dir, "com2sense_predictions.txt")
        pred_fo = open(pred_file, "w")
        for pred in preds:
            pred_fo.write(str(pred)+"\n")
        pred_fo.close()
        logging.info("Saving prediction file to: {}".format(pred_file))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False,
                            data_split="test", data_dir=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = data_processors[task](data_dir=args.data_dir, args=args)

    # Getting the examples.
    if data_split == "test" and evaluate:
        examples = (processor.get_test_examples())
    elif (data_split == "val" or data_split == "dev") and evaluate:
        examples = (processor.get_dev_examples())
    elif data_split == "train" and evaluate:
        examples = (processor.get_train_examples())
    elif "test" == data_split:
        examples = (processor.get_test_examples())
    else:
        examples = (
            processor.get_test_examples()
            if evaluate else processor.get_train_examples()
        )

    logging.info("Number of {} examples in task {}: {}".format(
        data_split, task, len(examples)))

    # Defines the dataset.
    dataset = data_classes[task](examples, tokenizer,
                                 max_seq_length=args.max_seq_length,
                                 seed=args.seed, args=args)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier() 
    
    return dataset


def main():
    torch.autograd.set_detect_anomaly(True)
    
    args = get_args()

    # Writes the prefix to the output dir path.
    if args.output_root is not None:
        args.output_dir = os.path.join(args.output_root, args.output_dir)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use "
            "--overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup distant debugging if needed.
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training.
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs.
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed "
        "training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Sets seed.
    set_seed(args)

    # Loads pretrained model and tokenizer.
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab.
        torch.distributed.barrier() 

    # Getting the labels
    processor = data_processors[args.task_name]()
    num_labels = processor.get_labels()

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    ##################################################
    # TODO: Please fill in the below to obtain the
    # `AutoConfig`, `AutoTokenizer` and some auto
    # model classes correctly. Check the documentation
    # for essential args. For the model, please write
    # an if-else statements that to use MLM model when
    # `training_phase` is `pretrain` otherwise use the
    # sequence classification model.

    # TODO: Huggingface configs.
    raise NotImplementedError("Please finish the TODO!")

    # TODO: Tokenizer.
    raise NotImplementedError("Please finish the TODO!")

    # TODO: Defines the model.
    if args.training_phase == "pretrain":
        raise NotImplementedError("Please finish the TODO!")
    else:
        raise NotImplementedError("Please finish the TODO!")

    # End of TODO.
    ##################################################

    # Loads models onto the device (gpu or cpu).
    model.to(args.device)
    print(model)
    args.model_type = config.model_type

    logger.info("Training/evaluation parameters %s", args)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("!!! Number of Params: {} M".format(count_parameters(model)/float(1000000)))

    # Training.
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name,
                                                tokenizer, data_split="train",
                                                evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Evaluation.
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(
                    args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
        else:
            assert args.iters_to_eval is not None, ("At least one"
                " of `iter_to_eval` or `eval_all_checkpoints` should be set.")
            checkpoints = []
            for iter_to_eval in args.iters_to_eval:
                checkpoints_curr = list(
                    os.path.dirname(c) for c in sorted(glob.glob(
                        args.output_dir + "/*-{}/".format(iter_to_eval)
                        + WEIGHTS_NAME, recursive=True))
                )
                checkpoints += checkpoints_curr

        logger.info("\n\nEvaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            logger.info("\n\nEvaluate checkpoint: %s", checkpoint)
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            ckpt_path = os.path.join(checkpoint, "pytorch_model.bin")
            model.load_state_dict(torch.load(ckpt_path))
            model.to(args.device)

            ##################################################
            # TODO: Make sure the eval_split is "test" if in
            # testing phase.
            pass  # This TODO does not require any actual
                  # implementations, just a reminder.
            # End of TODO.
            ##################################################

            result = evaluate(args, model, tokenizer, prefix=prefix, data_split=args.eval_split)
            result = dict((k + "_{}".format(global_step), v)
                           for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
