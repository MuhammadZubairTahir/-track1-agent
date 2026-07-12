"""
Track 1 — General-Purpose AI Agent
Reads /input/tasks.json, answers each task using either:
  - a local quantized model (free, zero token cost), or
  - the Fireworks API (costs tokens, only used when needed)
Writes /output/results.json before exiting.
"""

import json
import os
import re
import sys
import time
import traceback

from llama_cpp import Llama
import requests

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
LOCAL_MODEL_PATH = "/app/models/local-model.gguf"

# ---------------------------------------------------------------------------
# Environment variables (provided by the harness at evaluation time)
# ---------------------------------------------------------------------------
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS = os.environ.get("ALLOWED_MODELS", "")
ALLOWED_MODELS = [m.strip() for m in ALLOWED_MODELS.split(",") if m.strip()]

# Pick the first allowed model as our default remote model.
# (You can add smarter selection later — e.g. a bigger model for reasoning,
# a smaller one for simple formatting tasks — as long as it's in ALLOWED_MODELS.)
DEFAULT_REMOTE_MODEL = ALLOWED_MODELS[0] if ALLOWED_MODELS else None


# ---------------------------------------------------------------------------
# Local model (loaded once, reused for every task that doesn't need Fireworks)
# ---------------------------------------------------------------------------
_local_llm = None


def get_local_llm():
    global _local_llm
    if _local_llm is None:
        _local_llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=os.cpu_count() or 2,
            verbose=False,
        )
    return _local_llm


def ask_local_model(prompt: str, max_tokens: int = 256) -> str:
    llm = get_local_llm()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return out["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Remote model (Fireworks) — only called when the router decides it's needed
# ---------------------------------------------------------------------------
def ask_fireworks(prompt: str, max_tokens: int = 400) -> str:
    if not (FIREWORKS_API_KEY and FIREWORKS_BASE_URL and DEFAULT_REMOTE_MODEL):
        raise RuntimeError(
            "Fireworks not configured (missing FIREWORKS_API_KEY / "
            "FIREWORKS_BASE_URL / ALLOWED_MODELS) — cannot route to remote model."
        )

    url = FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEFAULT_REMOTE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Category classification (cheap, rule-based — no model call needed)
# ---------------------------------------------------------------------------
def classify_category(prompt: str) -> str:
    p = prompt.lower()

    if any(k in p for k in ["classify the sentiment", "sentiment of this review", "sentiment"]):
        return "sentiment"
    if any(k in p for k in ["summarize", "summarise", "one sentence", "condense"]):
        return "summarization"
    if any(k in p for k in ["extract all named entities", "named entities", "extract entities"]):
        return "ner"
    if any(k in p for k in ["has a bug", "fix the bug", "find and fix", "debug"]):
        return "code_debug"
    if any(k in p for k in ["write a python function", "write a function", "implement a function"]):
        return "code_gen"
    if re.search(r"\b(own|owns|different pet|each own|puzzle)\b", p) and any(
        k in p for k in ["who", "which"]
    ):
        return "logic"
    if any(k in p for k in ["%", "percent", "how many", "total", "remain"]) and any(
        ch.isdigit() for ch in p
    ):
        return "math"
    return "factual"


# Categories the local small model is usually reliable enough for.
# Everything else routes to Fireworks (costs tokens, but keeps accuracy up).
LOCAL_CAPABLE = {"sentiment", "summarization", "ner", "factual"}
REMOTE_PREFERRED = {"math", "logic", "code_debug", "code_gen"}


def route_and_answer(prompt: str) -> str:
    category = classify_category(prompt)

    if category in LOCAL_CAPABLE:
        try:
            return ask_local_model(prompt)
        except Exception:
            # Local model failed for some reason — fall back to Fireworks
            # rather than returning garbage.
            return ask_fireworks(prompt)

    # category in REMOTE_PREFERRED (harder reasoning / code tasks)
    return ask_fireworks(prompt)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt", "")
        try:
            answer = route_and_answer(prompt)
        except Exception as e:
            # Never let one bad task crash the whole run — record an error
            # answer instead so the rest of the batch still scores.
            traceback.print_exc()
            answer = f"[error generating answer: {e}]"

        results.append({"task_id": task_id, "answer": answer})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    start = time.time()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"Done in {time.time() - start:.1f}s", file=sys.stderr)
