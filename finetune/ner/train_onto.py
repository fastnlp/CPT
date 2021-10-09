import argparse
import os
import fastNLP
import fitlog
import torch
import torch.nn as nn
import copy
import transformers
from fastNLP import (
    AccuracyMetric,
    AdamW,
    Const,
    CrossEntropyLoss,
    FitlogCallback,
    Callback,
    Trainer,
    WarmupCallback,
    cache_results,
    DistTrainer,
    get_local_rank,
    CrossEntropyLoss,
    DataSet,
    Tester,
    Vocabulary,
)
from fastNLP.io import MsraNERLoader
from fastNLP.models import BiLSTMCRF
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    # BertModel,
    BartConfig,
    BertTokenizer,
    BertConfig,
    BertModel,
)
import torch.distributed as dist
from fastNLP.modules.decoder import ConditionalRandomField
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.modules.decoder.crf import allowed_transitions
import torch.nn.functional as F
from fastNLP import SpanFPreRecMetric
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from modeling_cpt import CPTModel

if "p" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["p"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
dist.init_process_group("nccl")


parser = argparse.ArgumentParser()
parser.add_argument("--ptm_name", help='path to pre-trained model checkpoint')
parser.add_argument("--dataset", default='', help='path to data dir')
parser.add_argument("--use_decoder", type=int, help='whether to use decoder')
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--n_epoch", type=int, default=30)
parser.add_argument("--batch_size", type=int, help='micro batch size')
parser.add_argument("--update_every", type=int, help='gradient accumulation steps')
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--wmup", default=0, type=float, help='warmup ratio')
parser.add_argument("--crf", type=int, default=1, help='whether to use crf')
parser.add_argument("--local_rank")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--DU_DG", type=int, default=0, help='whether to use both DU & DG')
args = parser.parse_args()


if "bart" not in args.ptm_name:
    args.use_decoder = 0

if args.ptm_name[-1] == "/":
    args.ptm_name = args.ptm_name[:-1]

model_name = args.ptm_name.split(r"/")[-1]

demo = False
if demo:
    cache_fn = f"dataset/cache/demo-{model_name}"
else:
    cache_fn = f"dataset/cache/onto-{model_name}-{args.max_length}"




@cache_results(cache_fn, _refresh=False)
def load_data(tokenizer_name):
    db = MsraNERLoader().load(args.dataset)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    pad_idx = tokenizer.pad_token_id
    db.apply(
        lambda line: tokenizer(
            " ".join(line["raw_chars"]),
            truncation=True,
            max_length=args.max_length,
        )["input_ids"],
        new_field_name="tok_idx",
        use_tqdm=True,
    )
    db.apply(lambda line: len(line["tok_idx"]), new_field_name="seq_len", use_tqdm=True)
    db.apply(
        lambda line: line["target"][: line["seq_len"] - 2],
        new_field_name="target",
        use_tqdm=True,
    )
    db.apply(
        lambda line: [1 for _ in line["tok_idx"]],
        new_field_name="attention_mask",
        use_tqdm=True,
    )
    db.apply(lambda line: len(line["target"]), new_field_name="seq_len", use_tqdm=True)

    for name, ds in db.iter_datasets():
        for ins in ds:
            assert len(ins["target"]) + 2 == len(ins["tok_idx"])

    target_voc = Vocabulary(padding=None, unknown=None)
    target_voc.from_dataset(db.get_dataset("train"), field_name="target")
    target_voc.index_dataset(
        db.get_dataset("train"),
        db.get_dataset("dev"),
        db.get_dataset("test"),
        field_name="target",
    )
    db.set_vocab(target_voc, "target")
    db.set_pad_val("target", db.get_vocab("target").to_index("O"))
    db.set_pad_val("tok_idx", pad_idx)
    db.set_pad_val("attention_mask", 0)
    db.set_input("tok_idx", "target", "seq_len", "attention_mask")

    db.set_target("target", "seq_len")

    return db, len(db.get_vocab("target"))


if get_local_rank() != 0:
    dist.barrier()

data_bundle, label_num = load_data(args.ptm_name)
print(data_bundle)


class base_model(nn.Module):
    def __init__(self, ptm, label_size, dropout, target_vocab):
        super(base_model, self).__init__()
        self.encoder = None
        self.decoder = None
        if "bart" in model_name:
            self.config = BartConfig().from_pretrained(ptm)
            self.embed = CPTModel.from_pretrained(ptm, config=self.config)
            self.encoder = self.embed.get_encoder()
            self.decoder = self.embed.get_decoder()
        else:

            self.config = BartConfig().from_pretrained(ptm)
            self.embed = CPTModel.from_pretrained(ptm, config=self.config)

        self.fc = nn.Linear(self.config.hidden_size, label_size)
        if args.DU_DG == 1:
            self.fc = nn.Linear(self.config.hidden_size * 2, label_size)
        self.dropout = nn.Dropout(dropout)
        trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)
        self.crf = ConditionalRandomField(
            label_size, include_start_end_trans=True, allowed_transitions=trans
        )

    def _forward(self, tok_idx, attention_mask, seq_len=None, target=None):
        if "bart" in model_name:
            feats = self.encoder(
                input_ids=tok_idx,
                attention_mask=attention_mask,
            ).last_hidden_state
            if args.use_decoder == 1:
                feats = self.embed(
                    input_ids=tok_idx,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )["decoder_hidden_states"][-1]

        else:

            feats = self.embed(
                input_ids=tok_idx, attention_mask=attention_mask, return_dict=True
            ).encoder_last_hidden_state
            encoder_ouptut = feats
            if args.use_decoder == 1:
                feats = self.embed(
                    input_ids=tok_idx,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )["decoder_hidden_states"][-1]
                if args.DU_DG == 1:
                    feats = torch.cat([encoder_ouptut, feats], dim=-1)


        feats = self.dropout(feats)
        feats = self.fc(feats)[:, 1:-1, :]
        logits = F.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            if args.crf == 0:
                pred = torch.argmax(logits, dim=-1)
            return {Const.OUTPUT: pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            if args.crf == 0:
                loss = CrossEntropyLoss().get_loss(
                    pred=feats, target=target, seq_len=seq_len
                )
            return {Const.LOSS: loss}

    def forward(self, tok_idx, attention_mask, seq_len, target):
        return self._forward(tok_idx, attention_mask, seq_len, target)

    def predict(self, tok_idx, attention_mask, seq_len):
        return self._forward(tok_idx, attention_mask, seq_len)


model = base_model(
    args.ptm_name,
    label_num,
    dropout=args.dropout,
    target_vocab=data_bundle.get_vocab("target"),
)

if get_local_rank() == 0:
    flag1 = [0]
    # flag1 = [1, 0]
    flag = sum(flag1)
    fitlog.debug(flag=flag)
    os.makedirs("logs/onto", exist_ok=True)
    fitlog.set_log_dir("logs/onto")
    fitlog.add_hyper(args)
    # fitlog.commit(__file__)
    fitlog.set_rng_seed()
    fitlog.add_other(model_name, "model_name")
    dist.barrier()

no_decay_params = ["bias", "LayerNorm", "layer_norm"]
params = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if (not any(nd in n for nd in no_decay_params)) and ("crf." not in n)
        ],
        "weight_decay": 1e-2,
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay_params)
        ],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if "crf." in n],
        "weight_decay": 0.0,
        "lr": 40 * args.lr,
    },
]

optimizer = AdamW(params, lr=args.lr)

metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab("target"))


class EvaluateCallback(Callback):
    r"""
    通过使用该Callback可以使得Trainer在evaluate dev之外还可以evaluate其它数据集，比如测试集。每一次验证dev之前都会先验证EvaluateCallback
    中的数据。
    """

    def __init__(self, data=None, tester=None):
        r"""
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用Trainer中的metric对数据进行验证。如果需要传入多个
            DataSet请通过dict的方式传入。
        :param ~fastNLP.Tester,Dict[~fastNLP.DataSet] tester: Tester对象, 通过使用Tester对象，可以使得验证的metric与Trainer中
            的metric不一样。
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(
                            f"{name} in tester is not a valid fastNLP.Tester."
                        )
                    self.testers["tester-" + name] = test
            if isinstance(tester, Tester):
                self.testers["tester-test"] = tester
            for tester in self.testers.values():
                setattr(tester, "verbose", 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(
                    value, DataSet
                ), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets["data-" + key] = value
        elif isinstance(data, DataSet):
            self.datasets["data-test"] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError(
                "Trainer has no dev data, you cannot pass extra DataSet to do evaluation."
            )

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(
                    data=data,
                    model=self.model,
                    batch_size=self.trainer.kwargs.get(
                        "dev_batch_size", self.batch_size / 2
                    ),
                    metrics=self.trainer.metrics,
                    verbose=0,
                    use_tqdm=self.trainer.test_use_tqdm,
                )
                self.testers[key] = tester
        self.best_eval_result = []

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    ###
                    assert len(eval_result) == 1
                    if len(self.best_eval_result) == 0:
                        self.best_eval_result.append(eval_result)
                    else:
                        best_result = self.best_eval_result[-1]
                        if (
                            best_result["SpanFPreRecMetric"]["f"]
                            < eval_result["SpanFPreRecMetric"]["f"]
                        ):
                            self.best_eval_result[-1] = eval_result
                    fitlog.add_best_metric(self.best_eval_result[-1], name=key)
                    ### 

                    self.logger.info("EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))

                except Exception as e:
                    self.logger.error(
                        "Exception happens when evaluate on DataSet named `{}`.".format(
                            key
                        )
                    )
                    raise e


tester = Tester(
    data=data_bundle.get_dataset("test"),
    metrics=metric,
    model=model,
    batch_size=args.batch_size,
)
callbacks_master = [
    FitlogCallback(data=data_bundle.get_dataset("test"), log_loss_every=50),
    EvaluateCallback(tester={"ts": tester}),
]
callbacks_all = (
    [WarmupCallback(schedule="linear", warmup=args.wmup)] if args.wmup > 0 else []
)


trainer = DistTrainer(
    train_data=data_bundle.get_dataset("train"),
    model=model,
    optimizer=optimizer,
    batch_size_per_gpu=args.batch_size,
    n_epochs=args.n_epoch,
    dev_data=data_bundle.get_dataset("dev"),
    metrics=metric,
    callbacks_master=callbacks_master,
    callbacks_all=callbacks_all,
    update_every=args.update_every,
    save_path=r"model/onto"
    # fp16="O1",
)

train = trainer.train()
