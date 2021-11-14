import argparse
import collections
import json
import os
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import torch
from preprocess.cmrc2018_evaluate import get_eval
# from pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from transformers import BertConfig, BertForQuestionAnswering
from transformers import BertTokenizer, AlbertTokenizer, AdamW
from tools import utils
from tools.pytorch_optimization import get_optimization, warmup_linear
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
from glob import glob
import fitlog

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTForQuestionAnswering, CPTConfig

@torch.no_grad()
def evaluate(model, args, eval_examples, eval_features, eval_file, device, global_steps, best_f1, best_em, best_f1_em, do_save=True):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.checkpoint_dir,
                                          "predictions_steps" + str(global_steps) + ".json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')
    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False, num_workers=2)

    model.eval()
    all_results = []
    print("Start evaluating")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        example_indices = batch[-1]
        batch = tuple(t.to(device) for t in batch[:-1])
        net_input = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            # 'token_type_ids': batch[2],
        }
        outputs = model(**net_input)
        batch_start_logits, batch_end_logits = outputs[:2]
        batch_start_logits = batch_start_logits.detach().cpu().tolist()
        batch_end_logits = batch_end_logits.detach().cpu().tolist()
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i]
            end_logits = batch_end_logits[i]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                        start_logits=start_logits,
                                        end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(eval_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    print(tmp_result)
    if do_save:
        with open(args.log_file, 'a') as aw:
            aw.write(json.dumps(tmp_result) + '\n')
        fitlog.add_metric(tmp_result, global_steps)

        if float(tmp_result['F1']) > best_f1:
            best_f1 = float(tmp_result['F1'])

        if float(tmp_result['EM']) > best_em:
            best_em = float(tmp_result['EM'])

        if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
            best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
            utils.torch_save_model(model, args.checkpoint_dir,
                                {'f1': float(tmp_result['F1']), 'em': float(tmp_result['EM'])}, max_save_num=1)

    model.train()

    return best_f1, best_em, best_f1_em

@torch.no_grad()
def test(model, args, eval_examples, eval_features, device, name):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.checkpoint_dir, 'cmrc2018_predict.json')
    output_nbest_file = output_prediction_file.replace('predict', 'nbest_predict')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        # segment_ids = segment_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask)
        batch_start_logits, batch_end_logits = outputs[:2]
        batch_start_logits = batch_start_logits.detach().cpu().tolist()
        batch_end_logits = batch_end_logits.detach().cpu().tolist()

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i]
            end_logits = batch_end_logits[i]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)
    return output_prediction_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training parameter
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--cls_mode', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.05)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument("--power", default=1.0, type=float, help='lr decay power')
    parser.add_argument('--seed', type=list, default=[123])
    parser.add_argument('--fp16', action='store_true', default=False)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_steps', type=float, default=200)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--do_train', type=int, default=1)
    parser.add_argument('--do_predict', type=int, default=0)
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # data dir
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--dev_dir1', type=str, required=True)
    parser.add_argument('--dev_dir2', type=str, required=True)
    parser.add_argument('--test_dir1', type=str, required=True)
    parser.add_argument('--test_dir2', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--init_restore_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--torch_fp16', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1)

    # use some global vars for convenience
    args = parser.parse_args()

    if args.task_name.lower() == 'drcd':
        from preprocess.DRCD_output import write_predictions
        from preprocess.DRCD_preprocess import json2features
    elif args.task_name.lower() == 'cmrc2018':
        from preprocess.cmrc2018_output import write_predictions
        from preprocess.cmrc2018_preprocess import json2features
    else:
        raise NotImplementedError

    args.train_dir = args.train_dir.replace('features.json', 'features_' + str(args.max_seq_length) + '.json')
    args.dev_dir1 = args.dev_dir1.replace('examples.json', 'examples_' + str(args.max_seq_length) + '.json')
    args.dev_dir2 = args.dev_dir2.replace('features.json', 'features_' + str(args.max_seq_length) + '.json')
    args = utils.check_args(args)
    device = torch.device("cuda")
    n_gpu = 1
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = dist.get_world_size()

    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.fp16 or args.torch_fp16))

    if args.local_rank > 0:
        torch.distributed.barrier()

    # load the bert setting
    if 'bert' in args.model_type or 'cpt' in args.model_type:
        if 'large' in args.init_restore_dir or '24' in args.init_restore_dir:
            config_path = 'hfl/chinese-roberta-wwm-ext-large'
        else:
            config_path = 'hfl/chinese-roberta-wwm-ext'
        bert_config = BertConfig.from_pretrained(config_path)
        tokenizer = BertTokenizer.from_pretrained(config_path)
        bert_config.hidden_dropout_prob = args.dropout
        bert_config.attention_probs_dropout_prob = args.dropout
        if 'cpt' in args.init_restore_dir:
            config = CPTConfig.from_pretrained(args.init_restore_dir)
            config.cls_mode = args.cls_mode
            config.attention_dropout = args.dropout
            config.activation_dropout = args.dropout
            config.dropout = args.dropout
            print(config)
            model = CPTForQuestionAnswering.from_pretrained(args.init_restore_dir, config=config)
        else:
            model_path = args.init_restore_dir
            model = BertForQuestionAnswering.from_pretrained(model_path, config=bert_config)

    # load data
    print('loading data...')
    
    if not os.path.exists(args.train_dir):
        json2features(args.train_file, [args.train_dir.replace('_features_', '_examples_'), args.train_dir],
                      tokenizer, is_training=True,
                      max_seq_length=args.max_seq_length)

    if not os.path.exists(args.dev_dir1) or not os.path.exists(args.dev_dir2):
        json2features(args.dev_file, [args.dev_dir1, args.dev_dir2], tokenizer, is_training=False,
                      max_seq_length=args.max_seq_length)

    args.test_dir1 = args.test_dir1.replace('examples.json', 'examples_' + str(args.max_seq_length) + '.json')
    args.test_dir2 = args.test_dir2.replace('features.json', 'features_' + str(args.max_seq_length) + '.json')

    if not os.path.exists(args.test_dir1) or not os.path.exists(args.test_dir2):
        json2features(args.test_file, [args.test_dir1, args.test_dir2], tokenizer, is_training=False,
                      max_seq_length=args.max_seq_length)
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_features = json.load(open(args.train_dir, 'r'))
    dev_examples = json.load(open(args.dev_dir1, 'r'))
    dev_features = json.load(open(args.dev_dir2, 'r'))
    test_examples = json.load(open(args.test_dir1, 'r'))
    test_features = json.load(open(args.test_dir2, 'r'))

    if os.path.exists(args.log_file) and args.local_rank in [-1, 0]:
        os.remove(args.log_file)

    steps_per_epoch = len(train_features) // args.n_batch
    eval_steps = args.eval_steps
    dev_steps_per_epoch = len(dev_features) // args.n_batch
    if len(train_features) % args.n_batch != 0:
        steps_per_epoch += 1
    if len(dev_features) % args.n_batch != 0:
        dev_steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs

    print('steps per epoch:', steps_per_epoch)
    print('total steps:', total_steps)
    print('warmup steps:', int(args.warmup_rate * total_steps))

    F1s = []
    EMs = []
    # 存一个全局最优的模型
    best_f1_em = 0
    best_f1, best_em = 0, 0
    seed_ = args.seed[0]
    with open(args.log_file, 'a') as aw:
        aw.write('===================================' +
                    'SEED:' + str(seed_)
                    + '===================================' + '\n')
    print('SEED:', seed_)

    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_)

    # init model
    print('init model...')
    utils.torch_show_all_params(model)
    model.to(device)

    if args.do_train:
    
        if args.local_rank >= 0:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)

        
        all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)

        seq_len = all_input_ids.shape[1]
        # 样本长度不能超过bert的长度限制
        assert seq_len <= bert_config.max_position_embeddings

        # true label
        all_start_positions = torch.tensor([f['start_position'] for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f['end_position'] for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, batch_size=args.n_batch // n_gpu, sampler=train_sampler, num_workers=2)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.train_epochs
        model, optimizer, scheduler = get_optimization(model, t_total, args.warmup_rate, args.weight_decay_rate, args.lr, 1e-8, args.power)
        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')



        if args.local_rank in [-1, 0]:
            os.makedirs('logs', exist_ok=True)
            fitlog.set_log_dir('logs')
            args_to_save = args.__dict__.copy()
            args_to_save['init_restore_dir'] = args_to_save['init_restore_dir'].split('/')[-1]
            fitlog.add_hyper(args_to_save)
        if args.torch_fp16:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()

        print('***** Training *****')
        model.train()
        global_steps = 1
        best_em = 0
        best_f1 = 0
        # if args.local_rank in [-1, 0]:
        #     best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, device,
        #                                             global_steps, best_f1, best_em, best_f1_em)
            
        for i in range(int(args.train_epochs)):
            print('Starting epoch %d' % (i + 1))
            iteration = 1
            with tqdm(total=steps_per_epoch, desc='Epoch %d' % (i + 1), disable=(args.local_rank not in [-1,0])) as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    net_input = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        # 'token_type_ids': batch[2],
                        'start_positions': batch[3],
                        'end_positions': batch[4]
                    }
                    if args.torch_fp16:
                        with autocast():
                            loss = model(**net_input)[0]
                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        loss = model(**net_input)[0]
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.torch_fp16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        optimizer.zero_grad()
                        global_steps += 1
                        pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item()*args.gradient_accumulation_steps)})
                        pbar.update(1)
                    
                        if (global_steps + 1) % eval_steps == 0:
                            best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, args.dev_file, device,
                                                            global_steps, best_f1, best_em, best_f1_em)
                            fitlog.add_best_metric({'F1':best_f1, 'EM':best_em})
                    

        if args.local_rank in [-1, 0]:
            best_f1, best_em, best_f1_em = evaluate(model, args, dev_examples, dev_features, args.dev_file, device,
                                                    global_steps, best_f1, best_em, best_f1_em)
            fitlog.add_best_metric({'F1':best_f1, 'EM':best_em})

        F1s.append(best_f1)
        EMs.append(best_em)

        # release the memory
        del optimizer
        # torch.cuda.empty_cache()

        print('Mean F1:', np.mean(F1s), 'Mean EM:', np.mean(EMs))
        print('Best F1:', np.max(F1s), 'Best EM:', np.max(EMs))
        with open(args.log_file, 'a') as aw:
            aw.write('Mean(Best) F1:{}({})\n'.format(np.mean(F1s), np.max(F1s)))
            aw.write('Mean(Best) EM:{}({})\n'.format(np.mean(EMs), np.max(EMs)))
        
    if args.do_predict:
        model_path = args.checkpoint_dir
        if model_path.endswith('.pth') or \
            model_path.endswith('.pt') or \
            model_path.endswith('.bin'):
            pass
        else:
            model_path = glob(model_path + '/*.pth')
            print(model_path)
            # assert len(model_path) == 1
            # for path in model_path:
            #     pass
            # model_path = model_path[0]
            for path in model_path:
                # utils.torch_init_model(model, path)
                model = model.from_pretrained(path)
                name = path.split('_')[-1]
                evaluate(model, args, dev_examples, dev_features, args.dev_file, device, global_steps=990099, best_f1=0, best_em=0, best_f1_em=0, do_save=False)
                output_file = test(model, args, test_examples, test_features, device, name)
                if args.task_name.lower() == 'drcd':
                    print(args.test_file)
                    print(output_file)
                    test_results = get_eval(args.test_file, output_file)
                    print('eval result for {}: {}'.format(name, test_results))
        # test(model, args, dev_examples, dev_features, device)
        
