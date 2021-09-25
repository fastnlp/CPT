import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import transformers
from seqeval.metrics import classification_report, f1_score
from transformers import (AutoConfig, AutoModel, BertTokenizer,BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback)
from transformers.trainer_utils import is_main_process

from model import CWSModel
from utils import DataTrainingArguments, ModelArguments, load_json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTModel

parser = argparse.ArgumentParser()
parser.add_argument("--bert_name",default='/path/to/model/',type=str)
parser.add_argument("--dataset", default="msr",type=str)
parser.add_argument("--lr",default=2e-5,type=float)
parser.add_argument("--batch_size",default='16',type=str)
parser.add_argument("--epoch",default='10',type=str)
parser.add_argument("--data_dir",default="/path/to/dataset/",type='str')
args = parser.parse_args()
arg_dict=args.__dict__

logger = logging.getLogger(__name__)

dataset_name=arg_dict['dataset']
outdir_1=arg_dict['bert_name'].split('/')[-1]
if not os.path.exists(outdir_1):
    os.mkdir(outdir_1)

outdir=outdir_1+'/'+dataset_name
if not os.path.exists(outdir):
    os.mkdir(outdir)

seed=len(os.listdir(outdir))+1
outdir=outdir+'/'+str(seed)

args=[
    '--model_name_or_path',arg_dict['bert_name'],
    '--do_train','--do_eval','--do_predict',
    '--train_file',os.path.join(arg_dict['data_dir'],'CWS.'+dataset_name,'train.json'),
    '--validation_file',os.path.join(arg_dict['data_dir'],'CWS.'+dataset_name,'dev.json'),
    '--test_file',os.path.join(arg_dict['data_dir'],'CWS.'+dataset_name,'test.json'),
    '--output_dir',outdir,
    '--per_device_train_batch_size',arg_dict['batch_size'],
    '--per_device_eval_batch_size',arg_dict['batch_size'],
    '--overwrite_output_dir',
    '--max_source_length=300',
    '--predict_with_generate=1',
    '--seed',str(1000*seed),
    '--num_train_epochs',arg_dict['epoch'],
    '--save_strategy','no',
    '--evaluation_strategy','epoch',
    '--learning_rate',str(arg_dict['lr']),
]
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(training_args.seed)

datasets={}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file
for key in data_files:
    print(key)
    datasets[key]=load_json(data_files[key])

print(datasets['train'][0])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
logger.info("Training/evaluation parameters %s", training_args)

label_list=['B','I','E','S']
label2idx={}
for i,label in enumerate(label_list):
    label2idx[label]=i
    

tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
config=AutoConfig.from_pretrained(model_args.model_name_or_path)
bart=CPTModel.from_pretrained(model_args.model_name_or_path)

model=CWSModel(encoder=bart.encoder,config=config)

def replace_unk(text):
    tokenized_text=tokenizer.encode(text,padding=False)
    while 100 in tokenized_text:
        idx=tokenized_text.index(100)
        text[idx-1]='[UNK]'
        tokenized_text[idx]=-1
    return text

def tokenize_and_align_labels(examples):
    for i in range(len(examples['text'])):
        examples['text'][i]=replace_unk(examples['text'][i])
    
    tokenized_inputs = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=data_args.max_source_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    
    labels = []   
    for i,label in enumerate(examples['labels']):
        label_t=list(map(lambda x:label2idx[x],label))[:data_args.max_source_length-2]
        label_t=[-100]+label_t+[-100]
        if len(label_t)!=len(tokenized_inputs['input_ids'][i]):
            print(i,len(label_t),len(tokenized_inputs['input_ids'][i]))
        assert(len(label_t)==len(tokenized_inputs['input_ids'][i]))
        labels.append(label_t)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = datasets['train'].map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
eval_dataset = datasets['validation'].map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
test_dataset=datasets['test'].map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
train_dataset=train_dataset.remove_columns('text')
eval_dataset=eval_dataset.remove_columns('text')
test_dataset=test_dataset.remove_columns('text')
 
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    metric=classification_report(y_true=true_labels,y_pred=true_predictions,output_dict=True,digits=4)
    metric=metric['micro avg']

    return metric

data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

class TestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        metrics['epoch']=state.epoch
        print(metrics)
        state.log_history.append(metrics)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TestCallback],
)

if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_metrics("train", metrics)
    trainer.save_state()