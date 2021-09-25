# Fine-tuning CPT for Sequence Classification

## Dataset
The dataset of **CLUE** can be downloaded [HERE](https://github.com/CLUEbenchmark/CLUE)

## Train and Evaluate
To train and evaluate **CPT$_u$**, **CPT$_g$** and **CPT$_{ug}$**, run the python file `run_clue_classifier.py`, with the argument `--cls_mode` be set to `1`, `2` and `3`, respectively. Following is a script example to run base version of **CPT$_u$** on **AFQMC** dataset.

```bash
export MODEL_TYPE=cpt-base
export MODEL_NAME=fnlp/cpt-base
export CLUE_DATA_DIR=/path/to/clue_data_dir
export TASK_NAME=afqmc
export CLS_MODE=1
python run_clue_classifier.py \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$MODEL_NAME \
    --cls_mode=$CLS_MODE \
    --task_name=$TASK_NAME \
    --do_train=True \
    --do_predict=1 \
    --no_tqdm=False \
    --data_dir=$CLUE_DATA_DIR/${TASK_NAME}/ \
    --max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --gradient_accumulation_steps 1 \
    --per_gpu_eval_batch_size=64 \
    --weight_decay=0.1 \
    --adam_epsilon=1e-6 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --max_grad_norm=1.0 \
    --learning_rate=1e-5 \
    --power=1.0 \
    --num_train_epochs=5.0 \
    --warmup_steps=0.1 \
    --logging_steps=200 \
    --save_steps=999999 \
    --output_dir=output/ft/$MODEL_TYPE/${TASK_NAME}/ \
    --overwrite_output_dir=True \
    --seed=42
```


## Prompt-based Fine-Tuning
To train and evaluate **CPT$_{u+p}$** and **CPT$_{g+p}$**, run the python file `run_clue_prompt.py` with the argument `--cls_mode` be set to `1` and `2`, respectively. Following is a script example to run base version of **CPT$_{u+p}$** on **AFQMC** dataset.

```bash
export MODEL_TYPE=cpt-base
export MODEL_NAME=fnlp/cpt-base
export CLUE_DATA_DIR=/path/to/clue_data_dir
export TASK_NAME=afqmc
export NUM_TRAIN=-1
export PATTERN_IDS=0
export CLS_MODE=1
python run_clue_prompt.py \
--pattern_ids $PATTERN_IDS \
--cls_mode 1 \
--data_dir=$CLUE_DATA_DIR/${TASK_NAME}/ \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME \
--max_seq_length 512 \
--task_name $TASK_NAME \
--output_dir output/prompt/$MODEL_TYPE/${TASK_NAME}/ \
--train_examples $NUM_TRAIN \
--weight_decay 0.1 \
--learning_rate 1e-5 \
--power 1.0 \
--warmup_steps 0.1 \
--split_examples_evenly \
--num_train_epochs 5 \
--eval_steps 200 \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--per_gpu_eval_batch_size 32 \
--do_train \
--do_eval
```
