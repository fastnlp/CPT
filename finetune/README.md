# Fine-Tuning CPT

This repo contains the fine-tuning code for CPT on multiple NLU and NLG tasks, such as text classification, machine reading comprehension (MRC), sequence labeling and text generation, etc.

## Requirements
- pytorch==1.8.1
- transformers==4.4.1
- fitlog
- fastNLP

## Run
The code and running examples are listed in the corresponding folders of the fine-tuning tasks.

- **`classification`**: [Fine-tuning](classification/README.md) for sequence classification with either external classifiers or prompt-based learning.
- **`cws`**: [Fine-tuning](cws/README.md) for Chinese Word Segmentation with external classifiers.
- **`generation`**: [Fine-tuning](generation/README.md) for abstractive summarization and data-to-text generation.
- **`mrc`**: [Fine-tuning](mrc/README.md) for Span-based Machine Reading Comprehension with exteranl classifiers.
- **`ner`**: [Fine-tuning](ner/README.md) for Named Entity Recognition.

You can also fine-tuning CPT on other tasks by adding `modeling_cpt.py` into your project and use the following code to use CPT.

```python
from modeling_cpt import CPTForConditionalGeneration
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = CPTForConditionalGeneration.from_pretrained("MODEL_NAME")
print(model)
```
