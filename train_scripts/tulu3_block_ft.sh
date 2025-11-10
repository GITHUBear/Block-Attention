export TOKENIZERS_PARALLELISM="false"
export WANDB_PROJECT="Block Attention For Tulu"
DS_CONFIG="configs/ds_stage2.json"


# The path of Tulu3-SFT epoch 2 checkpoint
MODEL_NAME="/data/shanhaikang.shk/model/blk_sft"

TRAIN_FP="datahub/mix_tulu3_rag.train"
# You can create a test set by yourself.
# EVAL_FP=""

LEARNING_RATE=2e-6

RUN_NAME="tulu_blk"

SAVE_DIR="/data/shanhaikang.shk/Block-Attention/tulu_save_dir"

mkdir -p $SAVE_DIR

SCRIPT_PATH=$(readlink -f "$0")

# cp $(realpath $EVAL_FP) $SAVE_DIR
cp $(realpath $TRAIN_FP) $SAVE_DIR
cp $SCRIPT_PATH $SAVE_DIR

deepspeed --num_gpus 8 trainer/hf_block_trainer.py \
  --model_name $MODEL_NAME \
  --max_length 4096 \
  --train_fp $TRAIN_FP \
  --dataloader_num_workers 16 \
  --dataloader_prefetch_factor 32 \
  --remove_unused_columns false \
  --do_train \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --train_prompt "false" \
  --train_full_attention "true" \
  --add_special_domain_tokens "true" \
  --loss_reduction "sum" \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.0 \
  --lr_scheduler_type "linear" \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --save_strategy "epoch" \
  --eval_strategy "no" \
  --logging_steps 1 \
  --bf16 \
  --optim "adamw_torch_fused" \
  --output_dir $SAVE_DIR \
  --logging_dir $SAVE_DIR \
  --run_name  $RUN_NAME \
  --report_to "wandb" \
  --deepspeed $DS_CONFIG
