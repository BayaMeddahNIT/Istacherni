import os
import sys
import time

# Set this to force tqdm to print to standard output even in environments that hide it
os.environ["TQDM_DISABLE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

print("="*60)
print("Starting explicit download of BAAI/bge-reranker-v2-m3...")
print("This model is ~2.2 GB. It may take 5-15 minutes depending on your internet.")
print("PLEASE DO NOT PRESS CTRL+C! Let it finish.")
print("="*60, flush=True)

try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        "BAAI/bge-reranker-v2-m3", 
        resume_download=True,
    )
    print(f"\nSUCCESS! Reranker model downloaded and cached at: {path}")
    print("You can now run your test scripts instantly.")
except Exception as e:
    print(f"\nError during download: {e}")
