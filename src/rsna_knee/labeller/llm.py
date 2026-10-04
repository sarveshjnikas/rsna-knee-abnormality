"""Call the LLM (synchronously or via the Batch API) and cache every raw answer on disk.

One request body is used for both paths, so sync and batch answers are comparable.
The cache is append-only JSONL keyed by study: re-runs never pay twice.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from rsna_knee.config import LABEL_DIR, ROOT
from rsna_knee.labeller.prompt import PROMPT_VERSION, SYSTEM_PROMPT, user_message
from rsna_knee.labeller.schema import JSON_SCHEMA

MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "low"  # lowest this model allows; reasoning tokens are billed as output
MAX_OUTPUT_TOKENS = 4000

# USD per 1M tokens: (input, cached input, output). Batch is half of standard.
PRICE_STANDARD = (2.00, 0.10, 10.00)
PRICE_BATCH = tuple(p / 2 for p in PRICE_STANDARD)

RUN_NAME = f"{PROMPT_VERSION}_{MODEL}"
CACHE_PATH = LABEL_DIR / f"raw_{RUN_NAME}.jsonl"
BATCH_LOG = LABEL_DIR / f"batches_{RUN_NAME}.json"

_cache_lock = threading.Lock()


def make_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    return OpenAI()


def request_body(report: str) -> dict:
    return {
        "model": MODEL,
        "reasoning": {"effort": REASONING_EFFORT},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(report)},
        ],
        "text": {"format": {"type": "json_schema", "name": "knee_findings", "strict": True, "schema": JSON_SCHEMA}},
    }


def output_text(response: dict) -> str:
    """The assistant's text from a Responses API object (raw dict, as in batch output)."""
    return "".join(
        part["text"]
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def cost_usd(usage: dict, price: tuple[float, float, float]) -> float:
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    uncached = usage["input_tokens"] - cached
    return (uncached * price[0] + cached * price[1] + usage["output_tokens"] * price[2]) / 1e6


# ------------------------------------------------------------------ cache


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["StudyInstanceUID"]: r for r in records}


def _append(record: dict, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock, path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(uid: str, response: dict, mode: str) -> dict:
    return {
        "StudyInstanceUID": uid,
        "mode": mode,
        "status": response.get("status"),
        "output_text": output_text(response),
        "usage": response.get("usage"),
    }


# ------------------------------------------------------------------ sync


def label_sync(studies: pd.DataFrame, workers: int = 8) -> list[dict]:
    """Label studies not yet cached, in parallel. Returns the new records."""
    cached = load_cache()
    todo = studies.loc[~studies["StudyInstanceUID"].isin(cached)]
    client = make_client()

    def one(row: pd.Series) -> dict:
        response = client.responses.create(**request_body(row["Report"])).model_dump()
        record = _record(row["StudyInstanceUID"], response, mode="sync")
        _append(record)
        return record

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, (row for _, row in todo.iterrows())))


# ------------------------------------------------------------------ batch


def _submit(client: OpenAI, chunk: pd.DataFrame, name: str) -> str:
    path = LABEL_DIR / f"batch_input_{RUN_NAME}_{name}.jsonl"
    with path.open("w") as f:
        for uid, report in zip(chunk["StudyInstanceUID"], chunk["Report"]):
            line = {"custom_id": uid, "method": "POST", "url": "/v1/responses", "body": request_body(report)}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    uploaded = client.files.create(file=path.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h")
    previous = json.loads(BATCH_LOG.read_text()) if BATCH_LOG.exists() else []
    BATCH_LOG.write_text(json.dumps(previous + [batch.id], indent=2))
    return batch.id


def _collect(client: OpenAI, batch) -> int:
    """Append a completed batch's answers to the cache; returns how many were new."""
    cached, added = load_cache(), 0
    for line in client.files.content(batch.output_file_id).text.splitlines():
        result = json.loads(line)
        uid = result["custom_id"]
        if uid in cached or result.get("error") or result["response"]["status_code"] != 200:
            continue
        _append(_record(uid, result["response"]["body"], mode="batch"))
        added += 1
    return added


def run_batches(studies: pd.DataFrame, chunk_size: int = 350, poll_seconds: int = 60) -> None:
    """Label all uncached studies with the Batch API, one chunk at a time.

    The organisation's queue holds at most ~900k enqueued tokens for this model,
    so chunks run sequentially; a chunk rejected for that limit is retried.
    """
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    while True:
        todo = studies.loc[~studies["StudyInstanceUID"].isin(load_cache())]
        if todo.empty:
            return
        chunk = todo.iloc[:chunk_size]
        batch_id = _submit(client, chunk, name=f"{int(time.time())}")
        while (batch := client.batches.retrieve(batch_id)).status in ("validating", "in_progress", "finalizing"):
            time.sleep(poll_seconds)
        if batch.status == "completed":
            print(f"{batch_id}: +{_collect(client, batch)} labelled, {len(todo) - len(chunk)} to go", flush=True)
        elif batch.errors and any(e.code == "token_limit_exceeded" for e in batch.errors.data):
            time.sleep(poll_seconds)  # an earlier batch still holds the queue
        else:
            raise RuntimeError(f"batch {batch_id} ended as {batch.status}: {batch.errors}")


def collect_batches() -> None:
    """Pull every completed batch in the log into the cache (safe to re-run)."""
    client = make_client()
    for batch_id in json.loads(BATCH_LOG.read_text()):
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            print(f"{batch_id}: +{_collect(client, batch)} new")
