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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa, CPT)."""


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    AlbertTokenizer,
    BertForSequenceClassification,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers.models.bert.tokenization_bert import BertTokenizer
from data_processors import clue_output_modes, clue_processors

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTForSequenceClassification, CPTConfig


processors.update(clue_processors)
output_modes.update(clue_output_modes)

GLOBAL_DATASETS = {}

import fitlog

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, power=1.0):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    if num_warmup_steps < 1:
        num_warmup_steps = int(num_training_steps * num_warmup_steps)
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        ) ** power

    return LambdaLR(optimizer, lr_lambda, last_epoch)

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()
    if args.local_rank in [-1, 0]:
        log_name = 'logs'
        os.makedirs(log_name, exist_ok=True)
        fitlog.set_log_dir(log_name)
        args_to_save = args.__dict__.copy()
        args_to_save['model_name_or_path'] = args.model_name_or_path.split('/')[-1]
        args_to_save.pop('device')
        args_to_save.pop('local_rank')
        args_to_save.pop('overwrite_cache')
        args_to_save.pop('keep_save_steps')
        args_to_save['cache_dir'] = ''
        fitlog.add_hyper(args_to_save)

    keep_save_steps = set(map(int, args.keep_save_steps.split()))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, collate_fn=train_dataset.collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "layer_norm", "layernorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total, power=args.power
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to global_step of last saved checkpoint from model path
    #     try:
    #         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     except ValueError:
    #         global_step = 0
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    best_metric = [0.0]
    def evaluate_during_train(best_metric):
        logs = {}
        dev_results = evaluate(args, model, tokenizer)
        for key, value in dev_results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value
        results = dev_results

        logger.info('eval at global_step: {}'.format(global_step))
        fitlog.add_metric(logs, global_step)
        save_best = False
        if results[args.main_metric] >= best_metric[0]:
            best_metric[0] = results[args.main_metric]
            fitlog.add_best_metric(best_metric[0], args.main_metric)
            save_best = True
        return logs, save_best

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0] or args.no_tqdm,
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0] or args.no_tqdm)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = {k: v.to(args.device) for k, v in batch.items()}

            # batch = tuple(t.to(args.device) for t in batch)
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs.pop('token_type_ids')
            outputs = model(**inputs, return_dict=False)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            # if isinstance(model, CPTForSequenceClassification):
            #     consist_loss = outputs[-1].item()
            #     enc_loss = outputs[-2].item()
            #     dec_loss = outputs[-3].item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                epoch_iterator.set_postfix(ordered_dict={
                    'loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                    'global_step': global_step,
                })

                do_save = False
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        _, do_save = evaluate_during_train(best_metric)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    # for key, value in logs.items():
                    #     tb_writer.add_scalar(key, value, global_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and (do_save
                                                   or (global_step in keep_save_steps)
                                                   or global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    if (
        args.local_rank in [-1, 0] and args.evaluate_during_training
    ):  # Only evaluate when single GPU otherwise metrics may not average well
        _, do_save = evaluate_during_train(best_metric)
        # print(json.dumps({**logs, **{"step": global_step}}))
        if (do_save or (global_step in keep_save_steps) \
                    or global_step % args.save_steps == 0):
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
    if args.local_rank in [-1, 0]:
        fitlog.finish()
    return global_step, tr_loss / global_step

@torch.no_grad()
def evaluate(args, model, tokenizer, prefix="", part='dev'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_metric_task = 'sst-2' # HACK: use acc for all tasks
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        eval_dataset = get_dataset(args, args.task_name, tokenizer, part)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=2, collate_fn=eval_dataset.collate_fn)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} for {} *****".format(prefix, part))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.no_tqdm):
            model.eval()
            # batch = tuple(t.to(args.device) for t in batch)

            inputs = {k: v.to(args.device) for k, v in batch.items()}
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs.pop('token_type_ids')
            outputs = model(**inputs, return_dict=False)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_metric_task, preds, out_label_ids)
        results.update(result)
        results['loss'] = eval_loss

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        logger.info(output_eval_file)
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def predict(args, model, tokenizer, label_list, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_metric_task = 'sst-2' # HACK: use acc for all tasks
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    label_map = {i: label for i, label in enumerate(label_list)}

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        eval_dataset = get_dataset(args, args.task_name, tokenizer, 'test')

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=2, collate_fn=eval_dataset.collate_fn)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        for batch in tqdm(eval_dataloader, desc="Predicting", disable=args.no_tqdm):
            model.eval()
            # batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {k: v.to(args.device) for k, v in batch.items()}
                # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                # if args.model_type != "distilbert":
                #     inputs["token_type_ids"] = (
                #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                inputs.pop('token_type_ids')
                outputs = model(**inputs, return_dict=False)
                tmp_eval_loss, logits = outputs[:2]

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            pred_labels = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            pred_labels = np.squeeze(preds)

        output_predict_file = os.path.join(eval_output_dir, prefix, "{}_predict.json".format(eval_task if eval_task != 'ocnli' else 'ocnli_50k'))
        logger.info(output_predict_file)
        with open(output_predict_file, "w") as writer:
            for i, pred in enumerate(pred_labels):
                json_d = {}
                json_d['id'] = i
                json_d['label'] = str(label_map[pred])
                writer.write(json.dumps(json_d) + '\n')
    return pred_labels

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            args.tokenizer_name,
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

from functools import partial
class ExampleDataset:
    def __init__(self, examples, tokenizer, label_list, max_length, output_mode, part='train'):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_label = True
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.output_mode = output_mode

    def label_from_example(self, example):
        if example.label is None:
            return None
        if self.output_mode == "classification":
            return self.label_map[example.label]
        elif self.output_mode == "regression":
            return float(example.label)
        raise KeyError(self.output_mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        data = self.tokenizer(text=item.text_a, text_pair=item.text_b, max_length=self.max_length, truncation=True)
        label = 0
        if self.has_label:
            label = self.label_from_example(item)
        data['label'] = label
        # return data, label
        return data

    def collate_fn(self, batch_items):
        batch_items = self.tokenizer.pad(batch_items, padding=True)
        batch_inputs = {}
        for k, v in batch_items.items():
            if k == 'label' and self.has_label:
                if self.output_mode == 'regression':
                    batch_inputs['labels'] = torch.tensor(v).float()
                else:
                    batch_inputs['labels'] = torch.tensor(v).long()
            else:
                batch_inputs[k] = torch.tensor(v, dtype=torch.long)
        return batch_inputs

    def collate_fn_v1(self, batch_items):
        data_items = [item[0] for item in batch_items]
        label_items = [item[1] for item in batch_items]
        batch_inputs = self.tokenizer.pad(data_items, padding=True)
        batch_inputs = {k:torch.tensor(v).long() for k,v in batch_inputs.items()}
        if self.has_label:
            if self.output_mode == 'regression':
                batch_inputs['labels'] = torch.tensor(label_items).float()
            else:
                batch_inputs['labels'] = torch.tensor(label_items).long()
        return batch_inputs

def get_dataset(args, task, tokenizer, part='train'):
    data_str = '{}_{}_{}_{}'.format(task, args.max_seq_length, tokenizer.name_or_path, part)
    if data_str in GLOBAL_DATASETS:
        return GLOBAL_DATASETS[data_str]

    if args.local_rank not in [-1, 0] and part == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    # logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    if part == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    elif part == 'test':
        examples = processor.get_test_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    if args.local_rank == 0 and part == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = ExampleDataset(examples, tokenizer, label_list, max_length=args.max_seq_length, output_mode=output_mode, part=part)
    GLOBAL_DATASETS[data_str] = dataset
    return dataset

def get_model(args, model_name_or_path, num_labels):
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if 'cpt' in args.model_type:
        config = CPTConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None)
        # config.consist_lambda = args.consist_lambda
        config.cls_mode = args.cls_mode
        model = CPTForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.config_name if args.config_name else model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=args.config_name if args.config_name else model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    return config, tokenizer, model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", type=str2bool, default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", type=int, default=0, help="Whether to run predict on the test set.")
    parser.add_argument(
        "--evaluate_during_training", type=str2bool, default=True, help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", type=str2bool, default=True, help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--power", default=1.0, type=float, help="The learning rate decay power for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=999999, help="Save checkpoint every X updates steps.")
    parser.add_argument("--keep_save_steps", type=str, default="", help="Save checkpoint every X updates steps.")
    parser.add_argument("--main_metric", type=str, default='acc', help="Main metric to compare")
    parser.add_argument(
        "--eval_all_checkpoints",
        type=str2bool, default=True,
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--cls_mode", default=1, type=int, help="CPT fine-tune `mode`")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--no_tqdm", type=str2bool, default=False, help="Avoid using tqdm when available")
    parser.add_argument("--sample_tokenize", type=str2bool, default=False, help="using sampling when tokenize")
    parser.add_argument("--freeze_embeddings", type=str2bool, default=False, help="freeze embedding params")


    parser.add_argument(
        "--overwrite_output_dir", type=str2bool, default=False, help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--train_out", type=str, default="")
    parser.add_argument("--train_log", type=str, default="")

    args = parser.parse_args()
    if args.train_out:
        args.output_dir = args.train_out

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # if args.task_name not in args.data_dir:
    #     args.data_dir = os.path.join(args.data_dir, args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config, tokenizer, model = get_model(args, args.model_name_or_path, num_labels)

    if args.freeze_embeddings:
        for param in list(model.get_input_embeddings().parameters()):
            param.requires_grad = False
        logger.info("Froze Embedding Layer")

    model.to(args.device)
    logger.info(model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        train_dataset = get_dataset(args, args.task_name, tokenizer, 'train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if False and args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            
        best_save_steps = max([int(path.split('-')[-1]) for path in checkpoints])

        training_args = torch.load(os.path.join(checkpoints[0], "training_args.bin"), map_location='cpu')
        args.max_seq_length = training_args.max_seq_length
        logger.info("Evaluate the following checkpoints: {}, max_length: {}".format(checkpoints, args.max_seq_length))
        keep_save_steps = args.keep_save_steps.split()
        print(keep_save_steps)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if  not args.do_predict==2 and int(global_step) != best_save_steps:
                continue
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            # if len(global_step) and len(keep_save_steps) and global_step not in keep_save_steps:
            #     continue
            # model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.load_state_dict(torch.load(checkpoint + '/pytorch_model.bin'))
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            if args.do_predict:
                predict(args, model, tokenizer, label_list, prefix=prefix)

    return results


if __name__ == "__main__":
    main()
