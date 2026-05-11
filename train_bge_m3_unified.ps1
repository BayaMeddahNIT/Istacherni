$env:CUDA_VISIBLE_DEVICES="0"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:USE_LIBUV="0"

Write-Host "Starting Unified Multi-head Fine-tuning for BGE-M3..."

python -m torch.distributed.run --nproc_per_node 1 `
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 `
    --model_name_or_path BAAI/bge-m3 `
    --train_data ./hard_negatives.jsonl `
    --output_dir ./models/bge-m3-unified `
    --learning_rate 2e-5 `
    --fp16 `
    --num_train_epochs 3 `
    --dataloader_drop_last True `
    --normlized True `
    --temperature 0.02 `
    --query_max_len 512 `
    --passage_max_len 512 `
    --train_group_size 2 `
    --negatives_cross_device `
    --logging_steps 5 `
    --same_task_within_batch True `
    --unified_finetuning True `
    --use_self_distill True `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 2 `
    --gradient_checkpointing True
