# Fine-tuning CPT for CWS

## Dataset
The dataset **MSR** and **PKU** is from **SIGHAN2005**, which can be downloaded [HERE](http://sighan.cs.uchicago.edu/bakeoff2005/).

## Train and Evaluate

To train and evaluate CPT on CWS dataset, run the python file `run_cws.py`. Following is a script example to run base version of **CPT$_u$** on **MSR** dataset.

```bash
export MODEL_TYPE=cpt-base
export MODEL_NAME=fnlp/cpt-base
export DATA_DIR=/path/to/cws_data_dir
python run_cws.py \
    --bert_name=$MODEL_NAME \
    --data_dir=$DATA_DIR \
    --dataset=msr \
    --lr=2e-5 \
    --batch_size=16 \
    --epoch=10 \
```