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
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy
from transformers import AdamW
import math
import string
from collections import OrderedDict
from transformers import BertTokenizer
from .utils import *

from . import log
logger = log.get_logger(__name__)

def read_jsonline(input_file):
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line.strip()))
    return lines

class PromptTemplate:
    MASK = '[mask]'
    TEXT_A = '[texta]'
    TEXT_B = '[textb]'
    SEP = '[sep]'

    def __init__(self, tokenizer, labels: List[str]):
        self.task = 'n/a'
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_map = {label:i for i,label in enumerate(labels)}
        self.questions = []
        self.gen_prompt = []
        self.label_words = []
        # self.label_word_ids = []
        self.label2label_words = []
        self.label_words_length = []
        self.weights = []
        self.mlm_positions = []
        self.need_remove_punc = []

    def remove_punct(self, text):
        return text.rstrip(string.punctuation)

    def truncate_tokens(self, token_a, token_b, need_to_truncate):
        if need_to_truncate == 0:
            return

        if token_b is None:
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
        
        if len(token_a) > len(token_b):
            trun_len_a = min(len(token_a) - len(token_b), need_to_truncate)
            need_to_truncate -= trun_len_a
            token_a = last_truncate_tokens(token_a, trun_len_a)
        else:
            trun_len_b = min(len(token_b) - len(token_a), need_to_truncate)
            need_to_truncate -= trun_len_b
            token_b = last_truncate_tokens(token_b, trun_len_b)
        
        if need_to_truncate == 0:
            return token_a, token_b

        trun_len_a = need_to_truncate // 2
        trun_len_b = need_to_truncate - trun_len_a
        token_a = last_truncate_tokens(token_a, trun_len_a)
        token_b = last_truncate_tokens(token_b, trun_len_b)
        return token_a, token_b

    def encode(self, template, text_a, text_b=None, pid=0, max_length=128):
        max_token_len = self.label_words_length[pid]
        need_remove_punc = self.need_remove_punc[pid]
        if need_remove_punc[0]:
            text_a = self.remove_punct(text_a)
        if text_b and need_remove_punc[1]:
            text_b = self.remove_punct(text_b)
        question = template.copy()
        p = question.copy()
        pos_of_texta = []
        pos_of_textb = []
        token_a = self.tokenizer(text_a, add_special_tokens=False)['input_ids']
        token_b = self.tokenizer(text_b, add_special_tokens=False)['input_ids'] if text_b else []
        prompt_length = 0
        for i, q in enumerate(question):
            if q == self.TEXT_A:
                p[i] = token_a
                pos_of_texta.append(i)
            elif q == self.TEXT_B:
                p[i] = token_b
                pos_of_textb.append(i)
            elif q == self.MASK:
                mask_tokens = [self.tokenizer.mask_token_id] * max_token_len
                p[i] = mask_tokens
                prompt_length += len(p[i])
            elif q == self.SEP:
                p[i] = [self.tokenizer.sep_token_id]
                prompt_length += 1
            else:
                p[i] = self.tokenizer(q, add_special_tokens=False)['input_ids']
                prompt_length += len(p[i])
        
        total_length = len(token_a) * len(pos_of_texta)
        if text_b:
            total_length += len(token_b) * len(pos_of_textb)
        total_length += prompt_length + 2
        
        assert len(pos_of_texta) <= 1
        assert len(pos_of_textb) <= 1

        if total_length > max_length:
            need_to_truncate = total_length - max_length
            token_a, token_b = self.truncate_tokens(token_a, token_b, need_to_truncate)
            for i in pos_of_texta:
                p[i] = token_a
            if text_b:
                for i in pos_of_textb:
                    p[i] = token_b
        
        def flatten_list(mylist):
            flat_list = []
            for i in mylist:
                flat_list += i
            return flat_list

        flat_p = [self.tokenizer.cls_token_id] + flatten_list(p) + [self.tokenizer.sep_token_id]

        return flat_p

    def add_template(self, question: List, label_words: List[str], gen_prompt=None, remove_punc=True, weight=1.0):
        if isinstance(remove_punc, bool):
            remove_punc = (remove_punc, remove_punc)
        assert isinstance(remove_punc, (tuple,list)) and len(remove_punc) == 2, remove_punc
        assert self.TEXT_A in question, f"{question}"
        assert len(label_words) == len(self.labels), label_words
        label_words = {label: word for (label, word) in zip(self.labels, label_words)}
        # start_pos = 0
        # count = 0
        # while start_pos < len(question):
        #     if question[start_pos] == self.MASK:
        #         count += 1
        #     start_pos += 1
        # assert count == 1, f'question: {question}, label_words: {label_words}, count: {count}'
        assert weight >= 0.0

        label2label_words = {}
        max_token_len = 1
        for label, token in label_words.items():
            tokenid = self.tokenizer.encode(token, add_special_tokens=False)
            if len(tokenid) > 1:
                # logger.warning(f'token:[{token}] is not a single piece for label:[{label}], tokenid:{tokenid}')
                max_token_len = max(max_token_len, len(tokenid))
            labelid = self.label_map[label]
            label2label_words[labelid] = tokenid
        
        label_pad_id = self.tokenizer.encode(['-'], add_special_tokens=False)[0]
        for labelid, tokenid in label2label_words.items():
            while len(tokenid) < max_token_len:
                tokenid.append(label_pad_id)

        logger.info(f'Has {len(label_words)} labels, max_token_len={max_token_len}')
        # label_word_ids = {tokenid: labelid for labelid, tokenid in label2label_words.items()}

        self.questions.append(question)
        self.gen_prompt.append(gen_prompt)
        self.label_words_length.append(max_token_len)
        self.label_words.append(label_words)
        # self.label_word_ids.append(label_word_ids)
        self.label2label_words.append(label2label_words)
        self.weights.append(weight)
        self.need_remove_punc.append(remove_punc)

    def __len__(self):
        return len(self.questions)

    @property
    def num_labels(self):
        return len(self.labels)

    # def get_lm_labels2labels(self, idx):
    #     return self.label_word_ids[idx]

    def get_labels2lm_labels(self, idx):
        return self.label2label_words[idx]

    def get_label_list(self):
        return self.labels

class RteTemplate(PromptTemplate):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, ["not_entailment", "entailment"])
        self.task = 'rte'
        label_words = ["No", "Yes"]

        # self.add_template(
        #     question="\"{textb}?\"{mask}, \"{texta}.\"",
        #     label_words=label_words)

        # self.add_template(
        #     question="{textb}? {mask}, {texta}.",
        #     label_words=label_words)

        # self.add_template(
        #     question="\"{textb}?\" {mask}. \"{texta}.\"",
        #     label_words=label_words)

        self.add_template(
            question="{textb}? {mask}. {texta}.",
            label_words=label_words)

        # self.add_template(
        #     question="{texta} Question: {textb} True or False? Answer: {mask}.",
        #     label_words=["false", "true"])

        self.add_template(
            question="{texta}. {mask}, I believe {textb}.",
            label_words=["Yet", 'Clearly'])
        # self.add_template(
        #     question="{texta}. {mask}, I think that {textb}.",
        #     label_words=['Meanwhile', 'Accordingly'])
        self.add_template(
            question="{texta}. {mask}, I think {textb}.",
            label_words=['Meanwhile', 'So'])

class MrpcTemplate(PromptTemplate):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, ["0", "1"]) # not equivalent, equivalent
        self.task = 'mrpc'
        self.add_template(
            question="{texta}. {mask}! {textb}",
            label_words=["Alas", "Rather"],
            remove_punc=(True, False)
        )
        self.add_template(
            question="{texta}. {mask}. This is the first time {textb}",
            label_words=['Thus', 'At'],
            remove_punc=(True, False)
        )
        self.add_template(
            question="{texta}. {mask}. That's right. {textb}",
            label_words=['Moreover', 'Instead'],
            remove_punc=(True, False)
        )

class Sst2Template(PromptTemplate):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, ["0", "1"]) # negative, positive
        self.task = 'sst-2'
        self.add_template(
            question="{texta} A {mask} one.",
            label_words = ['pathetic', 'irresistible'],
            remove_punc=False
        )

        self.add_template(
            question="{texta} A {mask} piece.",
            label_words = ['bad', 'wonderful'],
            remove_punc=False
        )

        self.add_template(
            question="{texta} All in all {mask}.",
            label_words = ['bad', 'delicious'],
            remove_punc=False
        )


class AfqmcTemplate(PromptTemplate):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, ['0', '1'])
        self.task = 'afqmc'
        self.add_template(
            question=['[texta]', '[mask]', '和' , '[textb]', '一致。'],
            label_words=['不', '是'],
        )

class TnewsTemplate(PromptTemplate):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, [str(i + 100) for i in range(17) if i not in [5, 11]])
        self.task = 'tnews'
        self.add_template(
            question=['新闻“', '[texta]', '”是一个' , '[mask]', '新闻。'],
            label_words=[
                '故事',
                '文化',
                '娱乐',
                '体育',
                '金融',
                '住房',
                '汽车',
                '教育',
                '科技',
                '军事',
                '旅游',
                '国际',
                '股票',
                '农业',
                '游戏',
            ],
        )

        self.add_template(
            question=['[texta]'],
            gen_prompt=['这是' , '[mask]', '新闻。'],
            label_words=[
                '故事',
                '文化',
                '娱乐',
                '体育',
                '金融',
                '住房',
                '汽车',
                '教育',
                '科技',
                '军事',
                '旅游',
                '国际',
                '股票',
                '农业',
                '游戏',
            ],
        )

class IflytekTemplate(PromptTemplate):
    def __init__(self, tokenizer, args):
        labels = []
        label_words = []
        for line in read_jsonline(os.path.join(args.data_dir, 'labels.json')):
            labels.append(line['label'])
            label_words.append(line['label_des'])
        super().__init__(tokenizer, labels)
        self.task = 'iflytek'
        self.add_template(
            question=['手机软件：', '[texta]', '是关于', '[mask]', '的。'],
            label_words=label_words
        )

        self.add_template(
            question=['[texta]'],
            gen_prompt=['这个手机软件用于', '[mask]'],
            label_words=label_words
        )


class CslTemplate(PromptTemplate):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, ['0', '1'])
        self.task = 'csl'
        self.add_template(
            question=['关键词：', '[texta]', '匹配摘要：', '[textb]', '?', '[mask]'],
            label_words=['否', '是']
        )
        

class CluewscTemplate(PromptTemplate):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, ['false', 'true'])
        self.task = 'wsc'
        self.add_template(
            # ['句中的指代关系:', '[texta]', '是', '[mask]', '的'],
            ['[texta]', '是', '[mask]', '的'],
            ['错', '对']
        )

class OcnliTemplate(PromptTemplate):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, ['contradiction', 'neutral', 'entailment'])
        self.task = 'ocnli'
        self.add_template(
            ['第一句：', '[texta]', '与第二句：', '[textb]', '的关系是：', '[mask]'],
            ['矛盾', '无关', '因果']
        )




TEMPLATES = {
    # 'rte': RteTemplate,
    # 'mrpc': MrpcTemplate,
    # 'sst-2': Sst2Template,
    'afqmc': AfqmcTemplate,
    'tnews': TnewsTemplate,
    'iflytek': IflytekTemplate,
    'csl': CslTemplate,
    'wsc': CluewscTemplate,
    'ocnli': OcnliTemplate,
}

CPT_TEMPLATES = {
    'afqmc': AfqmcTemplate,
    'tnews': TnewsTemplate,
    'iflytek': IflytekTemplate,
    'csl': CslTemplate,
    'wsc': CluewscTemplate,
    'ocnli': OcnliTemplate,
}
