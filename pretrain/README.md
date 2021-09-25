# Pre-Training CPT

The code of pre-training CPT is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). 

For **Setup**, **Data Processing** and **Training** of CPT, you can refer to the [README](README_megatron.md) of Megatron-LM. And the package [jieba_fast](https://github.com/deepcs233/jieba_fast) is needed for Whole Word Masking pre-training.

After processing the data, place the `.bin` and `.idx` files into `./dataset/`. And place vocab files into `vocab/bert_zh_vocab/`. Then, use the scripts `run_pretrain_bart.sh` and `run_pretrain_cpt.sh` to train Chinese BART and CPT, respectively.

## Main Changes
- Add `bart_model` and `cpt_model` for Megatron under `megatron/model`, to let Megatron can train on BART and CPT.
- Add `_HfAutoTokenizer` in `megatron/tokenizer/tokenizer.py` to let Megatron can use Tokenizers from Huggingface-Transformers. 
- Add `bart_dataset` and `cpt_dataset` under `megatron/data` to produce data for Whole Word Masking (WWM) and Denoising Auto-Encoder (DAE) pre-training.
- Add `tools/convert_ckpt.py` to convert Megatron checkpoints to Huggingface-Transformers format.
- Add `tools/preprocess_data.py` to preprocess and chunk large amount of text data into binary format used in Megatron.
