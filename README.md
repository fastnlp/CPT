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

