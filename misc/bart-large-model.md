# Chinese BART-Large

## Model description

This is a implementation of Chinese BART-Large.

## Usage

```python
>>> from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
>>> tokenizer = BertTokenizer.from_pretrained("fnlp/bart-large-chinese")
>>> model = BartForConditionalGeneration.from_pretrained("fnlp/bart-large-chinese")
>>> text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
>>> text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
    [{'generated_text': '北 京 是 中 华 人 民 共 和 国 的 首 都'}]
```

**Note: Please use BertTokenizer for the model vocabulary. DO NOT use original BartTokenizer.**

## Citation

```bibtex
@article{shao2021cpt,
  title={CPT: A Pre-Trained Unbalanced Transformerfor Both Chinese Language Understanding and Generation}, 
  author={Yunfan Shao and Zhichao Geng and Yitao Liu and Junqi Dai and Fei Yang and Li Zhe and Hujun Bao and Xipeng Qiu},
  journal={arXiv preprint arXiv:2109.05729},
  year={2021}
}
```

