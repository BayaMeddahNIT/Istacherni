@echo off
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set USE_LIBUV=0

echo Starting Unified Multi-head Fine-tuning for BGE-M3 (Single GPU)...

python -m torch.distributed.launch --nproc_per_node 1 --use_env ^
    --model_name_or_path BAAI/bge-m3 ^
    --train_data ./hard_negatives.jsonl ^
    --output_dir ./models/bge-m3-unified ^
    --learning_rate 2e-5 ^
    --fp16 ^
    --num_train_epochs 3 ^
    --dataloader_drop_last True ^
    --temperature 0.02 ^
    --query_max_len 512 ^
    --passage_max_len 512 ^
    --train_group_size 2 ^
    --negatives_cross_device ^
    --logging_steps 5 ^
    --unified_finetuning True ^
    --use_self_distill True ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 2 ^
    --gradient_checkpointing True
