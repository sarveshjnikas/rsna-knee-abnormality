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

BUDGET_USD = 15.0  # hard stop for all labelling spend (sync + batch, every prompt version)

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


def total_spend() -> tuple[float, int]:
    """USD spent so far across every cached run, and the number of batch answers it covers."""
    spent, n_batch = 0.0, 0
    for path in LABEL_DIR.glob("raw_*.jsonl"):
        for record in load_cache(path).values():
            if record.get("usage"):
                batch = record["mode"] == "batch"
                spent += cost_usd(record["usage"], PRICE_BATCH if batch else PRICE_STANDARD)
                n_batch += batch
    return spent, n_batch


ACTIVE = ("validating", "in_progress", "finalizing", "cancelling")


def _wait(client: OpenAI, batch_id: str, poll_seconds: int, stall_minutes: float = 4, nearly_done: float = 0.95):
    """Wait for a batch to end. A batch stuck near the end is cancelled: finished answers are kept,
    and the few stragglers are simply picked up by a later chunk."""
    last_count, last_change = -1, time.time()
    while (batch := client.batches.retrieve(batch_id)).status in ACTIVE:
        counts = batch.request_counts
        if counts.completed != last_count:
            last_count, last_change = counts.completed, time.time()
        stalled = time.time() - last_change > stall_minutes * 60
        if batch.status == "in_progress" and stalled and counts.total and counts.completed >= nearly_done * counts.total:
            print(f"{batch_id}: stalled at {counts.completed}/{counts.total}, cancelling to keep moving", flush=True)
            client.batches.cancel(batch_id)
        time.sleep(poll_seconds)
    return batch


def _wait_for_logged_batches(client: OpenAI, poll_seconds: int) -> None:
    """Finish and collect batches already in flight, so a restart never pays twice."""
    if not BATCH_LOG.exists():
        return
    for batch_id in json.loads(BATCH_LOG.read_text()):
        batch = _wait(client, batch_id, poll_seconds)
        if batch.output_file_id:
            _collect(client, batch)


def run_batches(studies: pd.DataFrame, chunk_size: int = 350, poll_seconds: int = 60) -> None:
    """Label all uncached studies with the Batch API, one chunk at a time.

    The organisation's queue holds at most ~900k enqueued tokens for this model,
    so chunks run sequentially; a chunk rejected for that limit is retried.
    Stops before any chunk that could take total spend past BUDGET_USD.
    """
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    _wait_for_logged_batches(client, poll_seconds)
    while True:
        todo = studies.loc[~studies["StudyInstanceUID"].isin(load_cache())]
        if todo.empty:
            return
        chunk = todo.iloc[:chunk_size]
        spent, n_batch = total_spend()
        per_study = spent / max(n_batch, 1) if n_batch else 0.005
        if spent + 1.5 * per_study * len(chunk) > BUDGET_USD:
            print(f"budget stop: spent ${spent:.2f}, next chunk would exceed ${BUDGET_USD:.0f}", flush=True)
            return
        batch_id = _submit(client, chunk, name=f"{int(time.time())}")
        batch = _wait(client, batch_id, poll_seconds)
        if batch.status in ("completed", "cancelled") and batch.output_file_id:
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
        if batch.output_file_id:
            print(f"{batch_id}: +{_collect(client, batch)} new")
