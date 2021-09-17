# CPT

This repository contains code and checkpoints for **CPT**.

[**CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation**](https://arxiv.org/pdf/2109.05729.pdf)

Yunfan Shao, Zhichao Geng, Yitao Liu, Junqi Dai, Fei Yang, Li Zhe, Hujun Bao, Xipeng Qiu

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

## Downloads & Usage

Coming soon.

## Chinese BART

We also provide a pre-trained Chinese BART as a byproduct. The BART models is pre-trained with the same corpora, tokenization and hyper-parameters of CPT.

#### Load with Huggingface-Transformers

Chinese BART is available in **base** and **large** versions, and can be loaded with Huggingface-Transformers. The example code is as follows, where `MODEL_NAME` is `fnlp/bart-base-chinese` or `fnlp/bart-large-chinese` for **base** or **large** size of BART, respectively.

```python
>>> tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
>>> model = BartForConditionalGeneration.from_pretrained("MODEL_NAME")
```

The checkpoints of Chinese BART can be downloaded here. 

- [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese): 6 layers encoder, 6 layers decoder, 12 heads and 768 model dim.
- [fnlp/bart-large-chinese](https://huggingface.co/fnlp/bart-large-chinese): 12 layers encoder, 12 layers decoder, 16 heads and 1024 model dim.



## Citation

```bibtex
@article{shao2021cpt,
  title={CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation}, 
  author={Yunfan Shao and Zhichao Geng and Yitao Liu and Junqi Dai and Fei Yang and Li Zhe and Hujun Bao and Xipeng Qiu},
  journal={arXiv preprint arXiv:2109.05729},
  year={2021}
}
```

