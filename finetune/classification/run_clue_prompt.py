from packaging.version import parse
import torch
import numpy as np
import argparse
import os
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, BertTokenizer

from prompt.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from prompt.utils import eq_div, set_seed
from prompt import prompt, templates, log
import json
from transformers import glue_processors, WEIGHTS_NAME
from data_processors import clue_processors

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTForMaskedLM

import glob

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def write_predictions(eval_output_dir, eval_task, label_list, pred_labels):
    label_map = {i: label for i, label in enumerate(label_list)}
    output_predict_file = os.path.join(eval_output_dir, "{}_predict.json".format(eval_task if eval_task != 'ocnli' else 'ocnli_50k'))
    logger.info(output_predict_file)
    with open(output_predict_file, "w") as writer:
        for i, pred in enumerate(pred_labels):
            json_d = {}
            json_d['id'] = i
            json_d['label'] = str(label_map[pred])
            writer.write(json.dumps(json_d) + '\n')

logger = log.get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, 
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET)")
    parser.add_argument("--reduction", default='wmean', choices=['wmean', 'mean'],
                        help="Reduction strategy for merging predictions from multiple PET models. Select either "
                             "uniform weighting (mean) or weighting based on train set accuracy (wmean)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")
    parser.add_argument("--no_distillation", action='store_true',
                        help="If set to true, no distillation is performed (only for PET)")
    parser.add_argument("--repetitions", default=1, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--power", default=1.0, type=float,
                        help="Decaying power for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=100,
                        help="evaluate every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--priming', action='store_true',
                        help="Whether to use priming for evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--task_domain", default='clue', choices=['clue', 'glue', 'other'])
    parser.add_argument("--cls_mode", type=int, default=1)
    parser.add_argument("--fp16", action='store_true')

    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
    #         and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_domain == 'glue':
        processor = glue_processors[args.task_name]()
    elif args.task_domain == 'clue':
        processor = clue_processors[args.task_name]()
    else:
        processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    args.pattern_ids = set(args.pattern_ids)
    assert len(args.pattern_ids) > 0

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET
    eval_data = load_examples(
        processor, args.data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)
    test_data = load_examples(
        processor, args.data_dir, 'test', num_examples=test_ex, num_examples_per_label=test_ex_per_label)

    set_seed(42)
    for repeat_time in range(args.repetitions):
        seed = np.random.randint(10000)
        args.seed = seed
        logger.info("=====SEED {}=====".format(args.seed))

        train_data = load_examples(
            processor, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label, seed=args.seed)

        args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

        args.device = 'cpu' if args.no_cuda else 'cuda'

        if 'yfshao' not in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
            prompts = templates.TEMPLATES[args.task_name](tokenizer, args)
            cls_mode = -1
        else:
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
            model = CPTForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, cls_mode=args.cls_mode)
            prompts = templates.CPT_TEMPLATES[args.task_name](tokenizer, args)
            cls_mode = args.cls_mode

        prompt.finetune(args, model, tokenizer, prompts, train_data, eval_data, cls_mode=cls_mode)
        checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
        best_save_steps = max([int(path.split('-')[-1]) for path in checkpoints])
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if int(global_step) != best_save_steps:
                continue
            model = model.from_pretrained(checkpoint)
            results = prompt.evaluate(args, model, tokenizer, prompts, eval_data, cls_mode=cls_mode)
            logger.info('Eval step {}, scores:{}'.format(global_step, results['scores']))
            results = prompt.evaluate(args, model, tokenizer, prompts, test_data, cls_mode=cls_mode)
            write_predictions(checkpoint, args.task_name, prompts.get_label_list(), results['predictions'])

        del model


if __name__ == "__main__":
    main()
