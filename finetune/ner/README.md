# Fine-tuning CPT for NER

## Requirement
To run NER code, please install the newest version of fastNLP on github.
```bash
pip install git+https://github.com/fastnlp/fastNLP
```

## Dataset
- **CLUENER** can be found [HERE](https://github.com/CLUEbenchmark/CLUE).
- **MSRA** can be downloaded by **fastNLP** automatically.
- **OntoNotes** can be downloaded [HERE](https://catalog.ldc.upenn.edu/LDC2011T03).

## Train and Evaluate
The example running scripts for training CPT on **MSRA** dataset are as follows.
```bash
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 \
    train_msra.py \
    --ptm_name fnlp/cpt-base \
    --dataset '' \
    --use_decoder 0 \
    --batch_size 16 \
    --update_every 1 
```