# scratch/cleanup_and_finalize.py
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RERANKER_DIR = PROJECT_ROOT / "reranker_finetuned"
CKPT_50 = RERANKER_DIR / "checkpoint-50"
CKPT_66 = RERANKER_DIR / "checkpoint-66"

def main():
    print("=" * 60)
    print("  Reranker Model Finalization & Disk Cleanup")
    print("=" * 60)
    
    # 1. Delete optimizer states to reclaim ~6.9 GB of disk space
    for name, path in [("checkpoint-50", CKPT_50), ("checkpoint-66", CKPT_66)]:
        if path.exists():
            opt_file = path / "optimizer.pt"
            if opt_file.exists():
                size_gb = opt_file.stat().st_size / (1024 ** 3)
                print(f"Deleting heavy optimizer file from {name} ({size_gb:.2f} GB)...")
                opt_file.unlink()
                print("Deleted.")
            
            rng_file = path / "rng_state.pth"
            if rng_file.exists():
                rng_file.unlink()
                print("Deleted rng_state.pth")
                
            scaler_file = path / "scaler.pt"
            if scaler_file.exists():
                scaler_file.unlink()
                print("Deleted scaler.pt")

            scheduler_file = path / "scheduler.pt"
            if scheduler_file.exists():
                scheduler_file.unlink()
                print("Deleted scheduler.pt")

    # 2. Copy the final model files from checkpoint-66 to reranker_finetuned (parent)
    if CKPT_66.exists():
        print("\nCopying model files from checkpoint-66 (epoch 3) to parent directory...")
        files_to_copy = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        for f_name in files_to_copy:
            src = CKPT_66 / f_name
            if src.exists():
                dest = RERANKER_DIR / f_name
                print(f"  Copying {f_name}...")
                shutil.copy2(src, dest)
        print("Model files successfully finalized in reranker_finetuned!")
        
    # 3. Clean up the checkpoint folders entirely if we're done
    print("\nCleaning up checkpoint-50 and checkpoint-66 directories to free up more space...")
    if CKPT_50.exists():
        shutil.rmtree(CKPT_50)
        print("Removed checkpoint-50.")
    if CKPT_66.exists():
        shutil.rmtree(CKPT_66)
        print("Removed checkpoint-66.")
        
    print("\nCleanup and model finalization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
