# Fine-tuning CPT for Sequence Classification

## Dataset
The dataset of **CMRC2018** can be downloaded [HERE](https://github.com/CLUEbenchmark/CLUE). And **DRCD** can be downloaded [HERE](https://github.com/DRCKnowledgeTeam/DRCD).

## Train and Evaluate
To train and evaluate **CPT$_u$**, **CPT$_g$** and **CPT$_{ug}$**, run the python file `run_mrc.py`, with the argument `--cls_mode` be set to `1`, `2` and `3`, respectively. Following is a script example to run base version of **CPT$_u$** on **DRCD** dataset.

```bash
export MODEL_TYPE=cpt-base
export MODEL_NAME=fnlp/cpt-base
export CLUE_DATA_DIR=~/workdir/datasets/CLUEdatasets/
export TASK_NAME=drcd
export CLS_MODE=1
python run_mrc.py \
  --fp16 \
  --model_type $MODEL_TYPE \
  --train_epochs=5 \
  --do_train=1 \
  --do_predict=1 \
  --n_batch=16 \
  --gradient_accumulation_steps 4 \
  --lr=3e-5 \
  --dropout=0.2 \
  --cls_mode=$CLS_MODE \
  --warmup_rate=0.1 \
  --weight_decay_rate=0.01 \
  --max_seq_length=512 \
  --eval_steps=200 \
  --task_name=$TASK_NAME \
  --init_restore_dir=$MODEL_NAME \
  --train_dir=$CLUE_DATA_DIR/$TASK_NAME/train_features.json \
  --train_file=$CLUE_DATA_DIR/$TASK_NAME/train.json \
  --dev_dir1=$CLUE_DATA_DIR/$TASK_NAME/dev_examples.json \
  --dev_dir2=$CLUE_DATA_DIR/$TASK_NAME/dev_features.json \
  --dev_file=$CLUE_DATA_DIR/$TASK_NAME/dev.json \
  --test_file=$CLUE_DATA_DIR/$TASK_NAME/test.json \
  --test_dir1=$CLUE_DATA_DIR/$TASK_NAME/test_examples_$MODEL_TYPE.json \
  --test_dir2=$CLUE_DATA_DIR/$TASK_NAME/test_features_$MODEL_TYPE.json \
  --checkpoint_dir=output/$MODEL_TYPE/$TASK_NAME/
```
