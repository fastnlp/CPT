# Chinese CPT-Large

## Model description

This is an implementation of CPT-Large. To use CPT, please import the file `modeling_cpt.py` (**Download** [Here](https://github.com/fastnlp/CPT/blob/master/finetune/modeling_cpt.py)) that define the architecture of CPT into your project.

[**CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation**](https://arxiv.org/pdf/2109.05729.pdf)

Yunfan Shao, Zhichao Geng, Yitao Liu, Junqi Dai, Fei Yang, Li Zhe, Hujun Bao, Xipeng Qiu

**Github Link:** https://github.com/fastnlp/CPT

## Usage

```python
>>> from modeling_cpt import CPTForConditionalGeneration
>>> from transformers import BertTokenizer
>>> tokenizer = BertTokenizer.from_pretrained("fnlp/cpt-large")
>>> model = CPTForConditionalGeneration.from_pretrained("fnlp/cpt-large")
>>> inputs = tokenizer.encode("北京是[MASK]的首都", return_tensors='pt')
>>> pred_ids = model.generate(input_ids, num_beams=4, max_length=20)
>>> print(tokenizer.convert_ids_to_tokens(pred_ids[i]))
    ['[SEP]', '[CLS]', '北', '京', '是', '中', '国', '的', '首', '都', '[SEP]']
```

**Note: Please use BertTokenizer for the model vocabulary. DO NOT use original BartTokenizer.**

## Citation

```bibtex
@article{shao2021cpt,
  title={CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation}, 
  author={Yunfan Shao and Zhichao Geng and Yitao Liu and Junqi Dai and Fei Yang and Li Zhe and Hujun Bao and Xipeng Qiu},
  journal={arXiv preprint arXiv:2109.05729},
  year={2021}
}
```

