"""
local_llm.py
------------
Adapter for local LLM generation via Ollama.
Ollama runs completely offline after the initial model download.

Why Ollama?
  - Runs fully on CPU (no GPU required)
  - Manages model downloads and serves them as a local REST API
  - Supports Arabic-capable models (mistral, qwen2, aya, etc.)
  - No API keys, no internet needed after setup

Setup (one-time):
  1. Download Ollama from https://ollama.com/download
  2. Run: ollama pull qwen2:7b    (best Arabic support, ~4GB)
     OR:  ollama pull mistral     (lighter option, ~4GB)
  3. Ollama runs automatically as a background service on Windows.

Usage:
  from graph_rag_local.local_llm import local_generate
  answer = local_generate(prompt)
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Model name — override in .env with LOCAL_LLM_MODEL
# Recommended: "qwen2:7b" (strong Arabic), "mistral" (lighter), "aya:8b" (Arabic-first)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2:7b")

# Ollama server (default local port)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ── Health check ───────────────────────────────────────────────────────────────
def is_ollama_running() -> bool:
    """Check if the local Ollama server is running."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_available_models() -> list[str]:
    """Return names of models already downloaded in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ── Repetition post-processor ──────────────────────────────────────────────────
def _strip_repetitions(text: str, min_block: int = 40) -> str:
    """
    Detect and remove repeated paragraphs from LLM output.
    Small models (0.5b–1.5b) tend to loop the same answer block multiple times.
    This keeps only the first occurrence of any repeated paragraph.
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen = []
    result = []
    for para in paragraphs:
        # Use only paragraphs long enough to matter for dedup
        key = para[:min_block].strip()
        if key and key in seen:
            break  # First repetition detected — stop here
        seen.append(key)
        result.append(para)
    return "\n\n".join(result).strip()


# ── Core generation ────────────────────────────────────────────────────────────
def local_generate(
    prompt: str,
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 512,   # 512 keeps qwen2:7b within the 5-min mobile timeout at ~2-3 tok/s
    stream: bool = False,
    timeout: int = None,   # None = wait indefinitely (required for slow CPU models)
    json_mode: bool = False, # Set to True to enforce JSON output (for tool calling)
) -> str:
    """
    Generate text using a local Ollama model.

    Args:
        prompt:       The full prompt string to send to the model.
        model:        Model name (defaults to LOCAL_LLM_MODEL from .env).
        temperature:  Sampling temperature (0.1 = focused and deterministic).
        max_tokens:   Maximum output tokens.
        stream:       If True, prints tokens as they arrive (nice for CLI).
        timeout:      Seconds before giving up on a request (default 180s).
        json_mode:    If True, forces the model to output valid JSON.

    Returns:
        The generated text as a string.

    Raises:
        ConnectionError: If Ollama is not running.
        RuntimeError:    If the model is not installed.
    """
    _model = model or LOCAL_LLM_MODEL

    if not is_ollama_running():
        raise ConnectionError(
            "Ollama is not running. Please start it:\n"
            "  1. Install Ollama: https://ollama.com/download\n"
            "  2. Open a terminal and run: ollama serve\n"
            "  3. Pull a model:  ollama pull qwen2:7b\n"
            "     OR:            ollama pull mistral"
        )

    available = list_available_models()
    if available:
        # Exact match: "qwen2:3b" must be in the list, not just "qwen2"
        # Normalise: Ollama sometimes stores "qwen2:3b" as "qwen2:3b" but may omit tags
        normalised_available = [m.split(":latest")[0].lower() for m in available]
        _model_lower = _model.lower()
        if _model_lower not in normalised_available and not any(_model_lower == m for m in normalised_available):
            raise RuntimeError(
                f"Model '{_model}' not found in Ollama.\n"
                f"Installed models: {available}\n"
                f"\nTo install, run:\n  ollama pull {_model}"
            )

    payload_dict = {
        "model": _model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 3072,        # Increased from 2048 to prevent cross-lingual hallucination (stability)
            "num_thread": 6,        # Reduced to prevent CPU saturation (allow OS cycles)
            "repeat_penalty": 1.15, # Prevent small models (0.5b) from looping answers
            "repeat_last_n": 128,   # Look-back window for repetition detection
            "stop": ["\n\n\n", "الإجابة:\nالإجابة:", "السؤال:"],  # Hard stop sequences
        },
    }
    if json_mode:
        payload_dict["format"] = "json"

    payload = json.dumps(payload_dict).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        if stream:
            # Stream tokens to stdout
            full_response = []
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode())
                        token = chunk.get("response", "")
                        print(token, end="", flush=True)
                        full_response.append(token)
                        if chunk.get("done"):
                            break
            print()  # newline at end
            return _strip_repetitions("".join(full_response).strip())
        else:
            # Non-streaming: return full response at once
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
            full_text = []
            for line in raw.strip().splitlines():
                try:
                    obj = json.loads(line)
                    full_text.append(obj.get("response", ""))
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            return _strip_repetitions("".join(full_text).strip())

    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Model '{_model}' returned 404. It may not be fully downloaded.\n"
                f"Run: ollama pull {_model}"
            ) from e
        raise


if __name__ == "__main__":
    print(f"Ollama running: {is_ollama_running()}")
    print(f"Available models: {list_available_models()}")
