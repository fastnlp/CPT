# CPT

This repository contains code and checkpoints for **CPT**.

[**CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation**](https://arxiv.org/pdf/2109.05729.pdf)

Yunfan Shao, Zhichao Geng, Yitao Liu, Junqi Dai, Fei Yang, Li Zhe, Hujun Bao, Xipeng Qiu

### News

**12/30/2022**

An updated version of CPT & Chinese BART are released. In the new version, we changed the following parts:

- **Vocabulary** We replace the old BERT vocabulary with a larger one of size 51271 built from the training data, in which we 1) add missing 6800+ Chinese characters (most of them are traditional Chinese characters); 2) remove redundant tokens (e.g.  Chinese character tokens with ## prefix); 3) add some English tokens to reduce OOV.
- **Position Embeddings** We extend the max_position_embeddings from 512 to 1024.

We initialize the new version of models with the old version of checkpoints with vocabulary alignment. Token embeddings found in the old checkpoints are copied. And other newly added parameters are randomly initialized. We further train the new CPT & Chinese BART 50K steps with batch size 2048, max-seq-length 1024, peak learning rate 2e-5, and warmup ratio 0.1.

The result compared to the previous checkpoints is as followings:

|            | AFQMC | IFLYTEK | CSL-sum | LCSTS |  AVG  |
| :--------- | :---: | :-----: | :-----: | :---: | :---: |
| Previous   |      |        |        |      |      |
| bart-base  | 73.0 |   60   |  62.1  | 37.8 | 58.23 |
| cpt-base   | 75.1 |  60.5  |  63.0  | 38.2 | 59.20 |
| bart-large | 75.7 |  62.1  |  64.2  | 40.6 | 60.65 |
| cpt-large  | 75.9 |  61.8  |  63.7  | 42.0 | 60.85 |
| Updataed   |      |        |        |      |      |
| bart-base  | 73.03 |  61.25  |  61.51  | 38.78 | 58.64 |
| cpt-base   | 74.40 |  61.23  |  62.09  | 38.81 | 59.13 |
| bart-large | 75.81 |  61.52  |  64.62  | 40.90 | 60.71 |
| cpt-large  | 75.97 |  61.63  |  63.83  | 42.08 | 60.88 |

The result shows that the updated models maintain comparative performance compared with previous checkpoints. There are still some cases that the updated model is slightly worse than the previous one, which results from the following reasons: 1) Training additional a few steps did not lead to significant performance improvement; 2) some downstream tasks are not affected by the newly added tokens and longer encoding sequences, but sensitive to the fine-tuning hyperparameters.

- Note that to use updated models, please update the  `modeling_cpt.py` (new version download [Here](https://github.com/fastnlp/CPT/blob/master/finetune/modeling_cpt.py)) and the vocabulary (refresh the cache).

## Introduction

Aiming to unify both NLU and NLG tasks, We propose a novel **C**hinese **P**re-trained Un-balanced **T**ransformer (**CPT**), which is an unbalanced Transformer encoder-decoder pre-trained with MLM and DAE jointly.

<p align="center">
	<br>
 	<img src="./misc\cpt-architecture-v1.png" width = "700" align=center />
	<br>
</p>

The architecture of CPT is a variant of the full Transformer and consists of three parts:

1. **Shared Encoder** (S-Enc): a Transformer encoder with fully-connected self-attention, which is designed to capture the common semantic representation for both language understanding and generation.
2. **Understanding Decoder** (U-Dec): a shallow Transformer encoder with fully-connected self-attention, which is designed for NLU tasks. The input of U-Dec is the output of S-Enc.
3. **Generation Decoder** (G-Dec): a Transformer decoder with masked self-attention, which is designed for generation tasks with auto-regressive fashion. G-Dec utilizes the output of S-Enc with cross-attention.

## Pre-Trained Models
We provide the pre-trained weights of CPT and Chinese BART with source code, which can be directly used in Huggingface-Transformers.

- **`Chinese BART-base`**: 6 layers Encoder, 6 layers Decoder, 12 Heads and 768 Model dim.
- **`Chinese BART-large`**: 12 layers Encoder, 12 layers Decoder, 16 Heads and 1024 Model dim.
- **`CPT-base`**: 10 layers S-Enc, 2 layers U-Dec/G-Dec, 12 Heads and 768 Model dim.
- **`CPT-large`**: 20 layers S-Enc, 4 layers U-Dec/G-Dec, 16 Heads and 1024 Model dim.

The pre-trained weights can be downloaded here.
| Model | `MODEL_NAME`|
| --- | --- |
| **`Chinese BART-base`**  | [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) | 
| **`Chinese BART-large`**   | [fnlp/bart-large-chinese](https://huggingface.co/fnlp/bart-large-chinese) |
| **`CPT-base`**   | [fnlp/cpt-base](https://huggingface.co/fnlp/cpt-base) | 
| **`CPT-large`**   | [fnlp/cpt-large](https://huggingface.co/fnlp/cpt-large) |

### Requirements:
- pytorch==1.8.1
- transformers==4.4.1

To use CPT, please import the file `modeling_cpt.py` (Download [Here](finetune/modeling_cpt.py)) that define the architecture of CPT into your project.
Then, use the PTMs as the following example, where `MODEL_NAME` is the corresponding  string that refers to the model.

For CPT:
```python
from modeling_cpt import CPTForConditionalGeneration
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = CPTForConditionalGeneration.from_pretrained("MODEL_NAME")
print(model)
```

For Chinese BART:
```python
from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BartForConditionalGeneration.from_pretrained("MODEL_NAME")
print(model)
```

After initializing the model, you can use the following lines to generate text.
```python
>>> input_ids = tokenizer.encode("北京是[MASK]的首都", return_tensors='pt')
>>> pred_ids = model.generate(input_ids, num_beams=4, max_length=20)
>>> print(tokenizer.convert_ids_to_tokens(pred_ids[0]))
    ['[SEP]', '[CLS]', '北', '京', '是', '中', '国', '的', '首', '都', '[SEP]']
```

## Pre-Training
Pre-training code and examples can be find [Here](pretrain/README.md).


## Fine-Tuning
Fine-tuning code and examples can be find [Here](finetune/README.md).


## Contact

If you have any problems, raise an issue or contact <yfshao@fudan.edu.cn>.

## Citation

```bibtex
@article{shao2021cpt,
  title={CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation}, 
  author={Yunfan Shao and Zhichao Geng and Yitao Liu and Junqi Dai and Fei Yang and Li Zhe and Hujun Bao and Xipeng Qiu},
  journal={arXiv preprint arXiv:2109.05729},
  year={2021}
}
```

