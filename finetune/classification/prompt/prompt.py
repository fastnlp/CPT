import ast
import json
import os
import random
import statistics
from abc import ABC
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Dict, List
from tqdm import trange, tqdm
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, ConcatDataset
import numpy as np
import torch
from transformers import glue_compute_metrics
from transformers import AdamW
import math

from .utils import *
from .templates import PromptTemplate

from . import log
logger = log.get_logger(__name__)


class Converter:
    def __init__(self, args, tokenizer, prompt: PromptTemplate):
        self.args = args
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.token_ids = None
        self.lm_label_ids = {}

    def convert_examples(self, examples: List[InputExample], pid: int) -> List[InputFeatures]:
        """convert input to mlm format, adding prompt, demonstration, etc"""
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.debug("Writing example {}".format(ex_index))
            example.idx = ex_index
            input_features = self.get_input_feature(example, pid)
            features.extend(input_features)
            if ex_index < 3:
                logger.debug(f'--- Example {ex_index} ---')
                for input_feature in input_features:
                    logger.debug(input_feature)
        logger.info(f'Processed {len(examples)} examples, got {len(features)} features')
        return features

    def truncate_input_example(self, example: InputExample, max_length):
        need_to_truncate = 0
        for pid in range(len(self.prompt)):
            input_ids = self.prompt.encode(self.prompt.questions[pid], example.text_a, example.text_b, pid)
            need_to_truncate = max(need_to_truncate, len(input_ids) - max_length)
        if need_to_truncate == 0:
            return
        token_a = self.tokenizer(example.text_a, add_special_tokens=False)['input_ids']
        if example.text_b is not None:
            token_b = self.tokenizer(example.text_b, add_special_tokens=False)['input_ids']
        else:
            token_b = []

        def last_truncate_tokens(tokens, num_truncate):
            if num_truncate <= 0:
                return tokens
            truncated_tokens = tokens[:-num_truncate]
            return truncated_tokens

        def random_truncate_tokens(tokens, num_truncate):
            if num_truncate <= 0:
                return tokens
            truncate_first = np.random.randint(num_truncate)
            truncate_last = num_truncate - truncate_first
            truncated_tokens = tokens[truncate_first:]
            if truncate_last > 0:
                truncated_tokens = tokens[:-truncate_last]
            return truncated_tokens

        def change_example(token_a, token_b, old_example):
            old_example.text_a = self.tokenizer.decode(token_a)
            if len(token_b):
                old_example.text_b = self.tokenizer.decode(token_b)
            return old_example
        
        if len(token_a) > len(token_b):
            trun_len_a = min(len(token_a) - len(token_b), need_to_truncate)
            need_to_truncate -= trun_len_a
            token_a = last_truncate_tokens(token_a, trun_len_a)
        else:
            trun_len_b = min(len(token_b) - len(token_a), need_to_truncate)
            need_to_truncate -= trun_len_b
            token_b = last_truncate_tokens(token_b, trun_len_b)
        
        if need_to_truncate == 0:
            return change_example(token_a, token_b, example)

        trun_len_a = need_to_truncate // 2
        trun_len_b = need_to_truncate - trun_len_a
        token_a = last_truncate_tokens(token_a, trun_len_a)
        token_b = last_truncate_tokens(token_b, trun_len_b)
        return change_example(token_a, token_b, example)

    def get_input_feature(self, example: InputExample, pid: int) -> InputFeatures:
        prompt = self.prompt.encode(self.prompt.questions[pid], example.text_a, example.text_b, pid, self.args.max_seq_length)
        data = InputFeatures(prompt)
        # if len(data.input_ids) > self.args.max_seq_length:
        #     self.truncate_input_example(example, self.args.max_seq_length)
        #     prompt = self.prompt.encode(example.text_a, example.text_b, pid)
        #     logger.debug('Trancating for example from {} to {}'.format(len(data.input_ids), len(prompt)))
        #     data = InputFeatures(prompt)
        if self.prompt.gen_prompt[pid] is not None:
            data.gen_prompt = self.prompt.encode(self.prompt.gen_prompt[pid], example.text_a, example.text_b, pid, self.args.max_seq_length)
        data.attention_mask = [1] * len(data.input_ids)
        data.idx = example.idx
        data.label = self.prompt.label_map[example.label]
        data.lm_label = self.prompt.get_labels2lm_labels(pid)[data.label]
        data.prompt_idx = pid
        mlm_positions = []
        mlm_text_ids = data.input_ids
        if data.gen_prompt is not None:
            mlm_text_ids = data.gen_prompt
        for i, token in enumerate(mlm_text_ids):
            if token == self.mask_id:
                mlm_positions.append(i)
        data.mlm_positions = mlm_positions
        return data

    # def get_lm_label_ids(self, prompt_idx: torch.Tensor) -> torch.Tensor:
    #     if self.token_ids is None or self.token_ids.shape[0] != len(self.prompt.label_word_ids):
    #         token_ids = [list(label_word_ids.keys()) for label_word_ids in self.prompt.label_word_ids]
    #         self.token_ids = torch.LongTensor(token_ids)
    #     self.token_ids = self.token_ids.to(prompt_idx)
    #     return self.token_ids[prompt_idx]

    def get_lm_label_ids(self, prompt_idx):
        if prompt_idx not in self.lm_label_ids:
            label2lm_label = self.prompt.get_labels2lm_labels(prompt_idx)
            lm_label_ids = []
            for i in range(len(label2lm_label)):
                lm_label = label2lm_label[i]
                lm_label_ids.append(lm_label)
            self.lm_label_ids[prompt_idx] = torch.LongTensor(lm_label_ids)
        # [n_label, k_mask]
        return self.lm_label_ids[prompt_idx]


    def convert_lm_logits_to_logits(self, prompt_idxs, lm_logits):
        """
        prompt_idx: [b]
        lm_logits: [b,k_mask,n_vocab]
        logits: [b, n_label]
        """
        n_labels = self.prompt.num_labels
        bsz = lm_logits.size(0)
        logits = torch.zeros((bsz, n_labels)).to(lm_logits)
        # import pdb; pdb.set_trace()
        for i, (idx, lm_logit) in enumerate(zip(prompt_idxs, lm_logits)):
            lm_label_ids = self.get_lm_label_ids(idx).to(lm_logits.device)
            # [k_mask, n_label]
            token_logit = batch_index_select(lm_logit, lm_label_ids.transpose(0,1))
            logits[i] = token_logit.sum(0)
        return logits

    def convert_labels_to_lm_labels(self, prompt_idx, labels, max_len):
        """
        prompt_idx: [b]
        labels: [b]
        lm_labels: [b, k_mask]
        """
        lm_labels = torch.full((labels.size(0), max_len), fill_value=self.pad_id).to(labels)
        for idx, label in zip(prompt_idx, labels):
            lm_label_ids = self.get_lm_label_ids(idx).to(labels)
            lm_label = lm_label_ids[label]
            lm_labels[idx][:lm_label_ids.size(1)] = lm_label
        return lm_labels

    # def convert_logits(self, logits: torch.Tensor, prompt_idx: torch.Tensor, mlm_positions: torch.Tensor) -> torch.Tensor:
    #     """convert mlm logits to classification logits, using label words, etc"""
    #     prompt_idx = prompt_idx.to(logits.device)
    #     mlm_positions = mlm_positions.to(logits.device)
    #     token_ids = self.get_token_ids(prompt_idx)
    #     mlm_logits = batch_index_select(logits, mlm_positions)
    #     task_logits = batch_index_select(mlm_logits, token_ids)
    #     return task_logits


def finetune(args, model, tokenizer, prompts: PromptTemplate, train_data, eval_data, cls_mode=-1):
    train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    per_gpu_train_batch_size = args.per_gpu_train_batch_size
    model.to(args.device)
    converter = Converter(args, tokenizer, prompts)

    eval_batch_size = args.per_gpu_eval_batch_size
    
    all_question_ids = []
    all_labels = []
    total_steps = int(math.ceil(len(train_data)/train_batch_size) * len(prompts) * args.num_train_epochs)
    if args.max_steps > 0:
        max_steps = total_steps
        total_steps = args.max_steps
    if args.warmup_steps > 1:
        num_warmup_steps = int(args.warmup_steps)
    else:
        assert args.warmup_steps >= 0
        num_warmup_steps = int(total_steps * args.warmup_steps)
    train_datasets = []
    # for pid in range(1):
    for pid in args.pattern_ids:
        train_dataset = ExampleDataset(train_data, tokenizer, converter, pid)
        train_datasets.append(train_dataset)
    total_train_dataset = ConcatDataset(train_datasets)
    logger.info('Train dataset length: {}'.format(len(total_train_dataset)))
    train_data_loader = DataLoader(total_train_dataset, 
        shuffle=True, batch_size=per_gpu_train_batch_size, num_workers=2, 
        collate_fn=train_datasets[0].collate_fn)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps, power=args.power)

    tr_loss = 0.0
    logging_loss = 0.0
    global_step = 0
    logging_steps = args.logging_steps
    eval_steps = args.eval_steps
    best_results = None
    with tqdm(total=total_steps, desc="Training") as pbar:
        while global_step < total_steps:
            for i, batch in enumerate(train_data_loader):
                model.train()
                if global_step < 1:
                    for j in range(min(5, batch['input_ids'].size(0))):
                        logger.debug('enc_tokens: %s' % tokenizer.convert_ids_to_tokens(batch['input_ids'][j]))
                        logger.debug('dec_tokens: %s' % tokenizer.convert_ids_to_tokens(batch['decoder_input_ids'][j]))
                        logger.debug('prompt_idx: {}, mlm_positions: {}, label: {}'.format(*[batch[n][j] for n in ['prompt_idx', 'mlm_positions', 'labels']]))
                        logger.debug(batch['prompt_idx'][j])
                        logger.debug(batch['mlm_positions'][j])
                        logger.debug(batch['labels'][j])
                        logger.debug(batch['lm_labels'][j])


                # labels = batch['labels']
                # indices = batch['idx'].numpy()
                # prompt_idx = batch['prompt_idx'].to(args.device)
                # token_ids = converter.get_token_ids(prompt_idx)
                # token_labels = batch_index_select(token_ids, labels.to(args.device))

                lm_labels = batch['lm_labels']
                lm_labels = lm_labels.to(args.device)

                if cls_mode == -1:
                    outputs = mlm_forward(model, batch, args.device)
                    logits = mlm_get_logits(outputs, batch, args)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), lm_labels.view(-1))
                else:
                    outputs = cpt_forward(model, batch, args.device)
                    logits = cpt_get_logits(outputs, batch, args, cls_mode)
                    if cls_mode == 3:
                        logits = []
                    loss = loss_fn(logits.view(-1, logits.size(-1)), lm_labels.view(-1))

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                single_loss = loss.item()
                tr_loss += single_loss


                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    pbar.update()
                    pbar.set_postfix({"loss": single_loss, "lr": scheduler.get_last_lr()[0], "step": global_step})

                    if global_step % eval_steps == 0:
                        results = evaluate(args, model, tokenizer, prompts, eval_data, cls_mode)
                        pbar.write("global_step[{}]:".format(global_step) + json.dumps(results['scores']))
                        if best_results is None or best_results['scores']['all']['acc'] < results['scores']['all']['acc']:
                            best_results = results
                            path = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                            logger.info('saving at {}'.format(path))
                            model.save_pretrained(path)

                if global_step >= total_steps:
                    break

    results = evaluate(args, model, tokenizer, prompts, eval_data, cls_mode)
    pbar.write("global_step[{}]:".format(global_step) + json.dumps(results['scores']))
    if best_results is None or best_results['scores']['all']['acc'] < results['scores']['all']['acc']:
        best_results = results
        model.save_pretrained(os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step)))


def mlm_forward(model, batch, device):
    inputs = {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask'],
    }
    if 'token_type_ids' in batch:
        inputs['token_type_ids'] = batch['token_type_ids']
    inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model(**inputs)
    return outputs

def cpt_forward(model, batch, device):
    inputs = {
        'input_ids': batch['input_ids'],
        'attention_mask': batch['attention_mask'],
        'decoder_input_ids': batch['decoder_input_ids'],
    }
    inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model(**inputs)
    return outputs

def mlm_get_logits(outputs, batch, args):
    mlm_positions = batch['mlm_positions'].to(args.device)
    logits = batch_index_select(outputs[0], mlm_positions)
    # if logits.dim() >= 3:
    #     # [b, l, h]
    #     logits = logits.mean(1)
    return logits

def gen_get_logits(outputs, batch, args):
    gen_positions = batch['mlm_positions'].to(args.device)
    logits = batch_index_select(outputs[0], gen_positions)
    # if logits.dim() >= 3:
    #     # [b, l, h]
    #     logits = logits.mean(1)
    return logits

def cpt_get_logits(outputs, batch, args, cls_mode=1):
    if cls_mode == 1:
        return mlm_get_logits([outputs[0][0]], batch, args)
    elif cls_mode == 2:
        return gen_get_logits([outputs[0][1]], batch, args)
    elif cls_mode == 3:
        enc_logits = mlm_get_logits([outputs[0][0]], batch, args)
        dec_logits = gen_get_logits([outputs[0][1]], batch, args)
        return (enc_logits + dec_logits)

def ensemble_logits_for_prediction(all_logits: np.ndarray, pooling='vote') -> np.ndarray:
    """
    all_logits: num_prompts x num_samples x num_labels
    return: num_samples x num_labels
    """
    if pooling == 'vote':
        all_preds = np.argmax(all_logits, axis=-1)
        logits = np.zeros((all_logits.shape[1:]))
        for i in range(logits.shape[-1]):
            logits[:, i] = np.sum(all_preds == i, axis=0)
    elif pooling == 'mean':
        pass
    elif pooling == 'max':
        pass
    return logits

def get_metric(predictions, metas, task_name, task_domain='clue'):
    if task_domain == 'clue':
        # use acc for clue
        metric = glue_compute_metrics('sst-2', predictions, metas['labels'])
    elif task_name not in ['qa']:
        metric = glue_compute_metrics(task_name, predictions, metas['labels'])
    elif task_name == 'em':
        metric = exact_match(predictions, metas['labels'], metas['question_ids'])
    else:
        raise ValueError(f"Metric '{metric}' not implemented")
    return metric

@torch.no_grad()
def evaluate(args, model, tokenizer, prompts: PromptTemplate, eval_data, cls_mode):
    model.to(args.device)
    converter = Converter(args, tokenizer, prompts)

    eval_batch_size = args.per_gpu_eval_batch_size
    
    all_labels = None
    total_steps = int(math.ceil(len(eval_data)/eval_batch_size) * len(prompts))
    eval_sampler = SequentialSampler(eval_data)

    with tqdm(total=total_steps, desc="Evaluating", disable=False) as pbar:
        model.eval()
        all_logits = []
        for pid in args.pattern_ids:
            eval_dataset = ExampleDataset(eval_data, tokenizer, converter, pid)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=2, collate_fn=eval_dataset.collate_fn)

            logits_per_pid = np.zeros((len(eval_dataset), prompts.num_labels))
            labels_per_pid = []
            for i, batch in enumerate(eval_dataloader):

                labels = batch['labels'].numpy()
                indices = batch['idx'].numpy()
                prompt_idx = batch['prompt_idx'].cpu()

                if cls_mode == -1:
                    outputs = mlm_forward(model, batch, args.device)
                    lm_logits = mlm_get_logits(outputs, batch, args).cpu()
                    logits = converter.convert_lm_logits_to_logits(prompt_idx, lm_logits)
                elif cls_mode == 1:
                    outputs = cpt_forward(model, batch, args.device)
                    lm_logits = cpt_get_logits(outputs, batch, args, cls_mode).cpu()
                    logits = converter.convert_lm_logits_to_logits(prompt_idx, lm_logits)
                elif cls_mode == 2:
                    input_ids = batch['decoder_input_ids']
                    mlm_positions = batch['mlm_positions']
                    encoder_outputs = model.get_encoder()(
                        batch['input_ids'].to(args.device), 
                        attention_mask=batch['attention_mask'].to(args.device),
                        return_dict=True)
                    lm_labels = converter.get_lm_label_ids(prompt_idx[0].item())
                    logits = torch.zeros((input_ids.size(0), lm_labels.size(0)))
                    seq_idx = torch.arange(0, lm_labels.size(1)).to(input_ids)
                    for i, lm_label in enumerate(lm_labels):
                        for j in range(input_ids.size(0)):
                            input_ids[j, mlm_positions[j] + 1] = lm_label
                        output = model(input_ids.to(args.device), encoder_outputs=encoder_outputs)
                        lm_logit = cpt_get_logits(output, batch, args, cls_mode).cpu()
                        # import pdb; pdb.set_trace()
                        logits[:, i] = lm_logit[:,seq_idx,lm_label].sum(1).squeeze(-1)
                else:
                    raise NotImplementedError

                logits_per_pid[indices] = logits.cpu()
                pbar.update()
                labels_per_pid.append(labels)

            all_logits.append(logits_per_pid)
            if all_labels is None:
                all_labels = np.concatenate(labels_per_pid)

    all_logits = np.stack(all_logits, axis=0)
    all_preds = ensemble_logits_for_prediction(all_logits, pooling='vote')
    logger.debug((all_logits.shape, all_preds.shape))

    results = {
        'logits': all_preds,
        'labels': all_labels,
        'raw_logits': all_logits,
    }

    predictions = np.argmax(results['logits'], axis=1)
    results['predictions'] = predictions
    scores = {}

    scores['all'] = get_metric(predictions, results, args.task_name, args.task_domain)
    for i, pid in enumerate(args.pattern_ids):
        pred_i = np.argmax(results['raw_logits'][i], axis=1)
        scores[f'single_{pid}'] = get_metric(pred_i, results, args.task_name, args.task_domain)

    results['scores'] = scores
    return results
