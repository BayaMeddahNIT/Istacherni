import os
import sys

# BAAI unified fine-tuning requires DDP. We simulate it for a single GPU:
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["USE_LIBUV"] = "0"

# Inject sys.argv for the FlagEmbedding module
sys.argv = [
    "run.py",
    "--model_name_or_path", "D:\\pfe_baya_models\\bge-m3-unified",
    "--train_data", "./hard_negatives.jsonl",
    "--output_dir", "D:\\pfe_baya_models\\bge-m3-unified",
    "--learning_rate", "1e-5",
    "--fp16",
    "--num_train_epochs", "2",
    "--dataloader_drop_last", "True",
    "--temperature", "0.02",
    "--query_max_len", "512",
    "--passage_max_len", "512",
    "--train_group_size", "2",
    "--negatives_cross_device",
    "--logging_steps", "5",
    "--save_strategy", "no",
    "--unified_finetuning", "True",
    "--use_self_distill", "True",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "2",
    "--gradient_checkpointing", "True",
    "--ddp_backend", "gloo"
]

# Run the official BGE-M3 training module directly
from FlagEmbedding.finetune.embedder.encoder_only.m3.__main__ import main
main()
