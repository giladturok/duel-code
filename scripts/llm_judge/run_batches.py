"""Submit / poll the two judge batches, or replay live as a fallback.

Usage
-----
  python run_batches.py submit    # submit both batches, write out/batch_ids.json
  python run_batches.py poll      # poll until ended, fetch + parse results,
                                  # retry refusals/errors once via live API
  python run_batches.py           # submit (if not yet submitted) then poll
  python run_batches.py live      # no-batch fallback: replay both request
                                  # files with AsyncAnthropic, semaphore 8

Results (blinded; unblinding happens only in analyze.py):
  out/abs_results.jsonl, out/pair_results.jsonl
      {"custom_id", "status": "ok"|"missing", "stop_reason", "parsed", ...}
  out/refusals.jsonl    every refusal seen (batch or live)
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import anthropic

# Output directory: default is the 110M study's out/. Set LLM_JUDGE_OUT to
# drive another study (e.g. .../out8b for the 8B prefix-grid batches).
# Set LLM_JUDGE_REQ_PREFIX (e.g. "ct_") to drive an add-on request set in
# the same directory: reads {prefix}abs_requests.jsonl etc., writes
# {prefix}abs_results.jsonl, {prefix}batch_ids.json, {prefix}refusals.jsonl.
OUT_DIR = os.environ.get("LLM_JUDGE_OUT", C.OUT_DIR)
REQ_PREFIX = os.environ.get("LLM_JUDGE_REQ_PREFIX", "")

ARMS = {
    "abs": os.path.join(OUT_DIR, f"{REQ_PREFIX}abs_requests.jsonl"),
    "pair": os.path.join(OUT_DIR, f"{REQ_PREFIX}pair_requests.jsonl"),
}
BATCH_IDS_PATH = os.path.join(OUT_DIR, f"{REQ_PREFIX}batch_ids.json")
REFUSALS_PATH = os.path.join(OUT_DIR, f"{REQ_PREFIX}refusals.jsonl")
POLL_SECONDS = 60
LIVE_CONCURRENCY = 8


def _log_refusal(record):
    with open(REFUSALS_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _extract(message):
    """(stop_reason, parsed_json_or_None, raw_text_or_None) from a Message."""
    stop_reason = getattr(message, "stop_reason", None)
    text = None
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text" and block.text:
            text = block.text
            break
    parsed = None
    if text is not None and stop_reason not in ("refusal",):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    return stop_reason, parsed, text


def _live_call(client, params):
    return client.messages.create(**params)


def submit(client):
    if os.path.exists(BATCH_IDS_PATH):
        ids = C.load_json(BATCH_IDS_PATH)
        print(f"batches already submitted: {ids}")
        return ids
    ids = {}
    for arm, path in ARMS.items():
        requests = [{"custom_id": r["custom_id"], "params": r["params"]}
                    for r in C.load_jsonl(path)]
        batch = client.messages.batches.create(requests=requests)
        ids[arm] = batch.id
        print(f"submitted {arm}: {len(requests)} requests -> batch {batch.id}")
    C.dump_json(ids, BATCH_IDS_PATH)
    print(f"wrote {BATCH_IDS_PATH}")
    return ids


def poll(client):
    ids = C.load_json(BATCH_IDS_PATH)
    pending = dict(ids)
    while pending:
        for arm, bid in list(pending.items()):
            batch = client.messages.batches.retrieve(bid)
            counts = batch.request_counts
            print(f"[{time.strftime('%H:%M:%S')}] {arm} {bid}: "
                  f"{batch.processing_status} "
                  f"(processing={counts.processing} "
                  f"succeeded={counts.succeeded} errored={counts.errored} "
                  f"expired={counts.expired})")
            if batch.processing_status == "ended":
                del pending[arm]
        if pending:
            time.sleep(POLL_SECONDS)
    for arm, bid in ids.items():
        fetch_arm(client, arm, bid)


def fetch_arm(client, arm, batch_id):
    """Fetch batch results (arbitrary order -> key by custom_id), retry
    refusals/errored/expired once via the live API, write results jsonl."""
    requests_by_id = {r["custom_id"]: r["params"]
                      for r in C.load_jsonl(ARMS[arm])}
    results = {}
    retry_ids = []
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        rtype = result.result.type
        if rtype == "succeeded":
            msg = result.result.message
            stop_reason, parsed, text = _extract(msg)
            if stop_reason == "refusal":
                _log_refusal({"custom_id": cid, "source": "batch",
                              "arm": arm, "raw_text": text})
                retry_ids.append(cid)
            elif parsed is None:
                results[cid] = {"custom_id": cid, "status": "missing",
                                "stop_reason": stop_reason,
                                "note": "unparseable", "raw_text": text}
            else:
                results[cid] = {"custom_id": cid, "status": "ok",
                                "stop_reason": stop_reason, "parsed": parsed}
        elif rtype in ("errored", "expired"):
            err = getattr(result.result, "error", None)
            results[cid] = {"custom_id": cid, "status": "missing",
                            "note": rtype,
                            "error": str(err) if err else None}
            retry_ids.append(cid)
        else:  # canceled
            results[cid] = {"custom_id": cid, "status": "missing",
                            "note": rtype}

    # One live retry for refusals / errored / expired.
    for cid in retry_ids:
        params = requests_by_id.get(cid)
        if params is None:
            continue
        try:
            msg = _live_call(client, params)
        except anthropic.APIError as e:
            results[cid] = {"custom_id": cid, "status": "missing",
                            "note": "live_retry_failed", "error": str(e)}
            continue
        stop_reason, parsed, text = _extract(msg)
        if stop_reason == "refusal" or parsed is None:
            if stop_reason == "refusal":
                _log_refusal({"custom_id": cid, "source": "live_retry",
                              "arm": arm, "raw_text": text})
            results[cid] = {"custom_id": cid, "status": "missing",
                            "stop_reason": stop_reason,
                            "note": "refused_or_unparseable_after_retry"}
        else:
            results[cid] = {"custom_id": cid, "status": "ok",
                            "stop_reason": stop_reason, "parsed": parsed,
                            "note": "live_retry"}

    # Requests that never came back at all.
    for cid in requests_by_id:
        if cid not in results:
            results[cid] = {"custom_id": cid, "status": "missing",
                            "note": "absent_from_batch_results"}

    out_path = os.path.join(OUT_DIR, f"{REQ_PREFIX}{arm}_results.jsonl")
    with open(out_path, "w") as fh:
        for cid in sorted(results):
            fh.write(json.dumps(results[cid]) + "\n")
    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"{arm}: {n_ok}/{len(requests_by_id)} ok -> {out_path}")


async def live_arm(arm):
    """No-batch fallback: replay a request file with AsyncAnthropic."""
    requests = C.load_jsonl(ARMS[arm])
    client = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(LIVE_CONCURRENCY)
    results = {}

    async def one(req):
        cid = req["custom_id"]
        async with sem:
            try:
                msg = await client.messages.create(**req["params"])
            except anthropic.APIError as e:
                results[cid] = {"custom_id": cid, "status": "missing",
                                "note": "live_failed", "error": str(e)}
                return
        stop_reason, parsed, text = _extract(msg)
        if stop_reason == "refusal":
            _log_refusal({"custom_id": cid, "source": "live", "arm": arm,
                          "raw_text": text})
            results[cid] = {"custom_id": cid, "status": "missing",
                            "stop_reason": stop_reason, "note": "refusal"}
        elif parsed is None:
            results[cid] = {"custom_id": cid, "status": "missing",
                            "stop_reason": stop_reason,
                            "note": "unparseable", "raw_text": text}
        else:
            results[cid] = {"custom_id": cid, "status": "ok",
                            "stop_reason": stop_reason, "parsed": parsed}
        if len(results) % 100 == 0:
            print(f"{arm}: {len(results)}/{len(requests)} done")

    await asyncio.gather(*(one(r) for r in requests))
    out_path = os.path.join(OUT_DIR, f"{REQ_PREFIX}{arm}_results.jsonl")
    with open(out_path, "w") as fh:
        for cid in sorted(results):
            fh.write(json.dumps(results[cid]) + "\n")
    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"{arm}: {n_ok}/{len(requests)} ok -> {out_path}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "live":
        for arm in ARMS:
            asyncio.run(live_arm(arm))
        return
    client = anthropic.Anthropic()
    if cmd == "submit":
        submit(client)
    elif cmd == "poll":
        poll(client)
    elif cmd == "run":
        submit(client)
        poll(client)
    else:
        raise SystemExit(f"unknown command {cmd!r} (submit|poll|run|live)")


if __name__ == "__main__":
    main()
