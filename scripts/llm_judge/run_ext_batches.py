"""Chunked submit + poll + fetch for the n=1000 judge-expansion batches.

Why this exists: the ext request files are 24-134 MB, and a single
batches.create with a >80 MB body dies at the API edge (observed: 502 Bad
Gateway; 'peer closed connection' mid-upload). This script submits each arm
in chunks of CHUNK_SIZE requests (~20 MB), with crash-safe incremental state
and phantom-batch adoption, then polls and writes results in the exact
schema of run_batches.py (ext_{arm}_results.jsonl, ext_refusals.jsonl).

Phantom adoption: a failed create may still have created the batch server-
side (the error can occur while reading the response). Before submitting
anything, recent batches are listed; any non-canceled batch whose total
request count exactly matches an unrecorded arm/chunk is ADOPTED instead of
resubmitted. Arm totals (7200/21600/6970/21582) are unique vs all previous
studies' batch sizes, so adoption by size is unambiguous. Within an arm,
chunks are submitted one at a time with max_retries=0, so at most one
ambiguous chunk can exist at any moment; it is re-identified by size on the
next run.

Usage (needs ANTHROPIC_API_KEY in env; run under the smdm conda env):
  python run_ext_batches.py run       # adopt/submit everything, poll, fetch
  python run_ext_batches.py inspect   # just list recent batches
State: {out_dir}/ext_batches_state.json  ({arm: [batch_id, ...]}).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK_SIZE = 3000
POLL_SECONDS = 120

STUDIES = {
    "110m": os.path.join(HERE, "out"),
    "8b": os.path.join(HERE, "out8b_nuc"),
}
ARMS = ("abs", "pair")


def req_path(out_dir, arm):
    return os.path.join(out_dir, f"ext_{arm}_requests.jsonl")


def state_path(out_dir):
    return os.path.join(out_dir, "ext_batches_state.json")


def load_state(out_dir):
    p = state_path(out_dir)
    return C.load_json(p) if os.path.exists(p) else {a: [] for a in ARMS}


def save_state(out_dir, state):
    C.dump_json(state, state_path(out_dir))


def batch_total(b):
    c = b.request_counts
    return (c.processing + c.succeeded + c.errored + c.canceled + c.expired)


def chunks_of(requests):
    return [requests[i:i + CHUNK_SIZE]
            for i in range(0, len(requests), CHUNK_SIZE)]


def recent_batches(client, limit=100):
    out = []
    for b in client.messages.batches.list(limit=min(limit, 100)):
        out.append(b)
        if len(out) >= limit:
            break
    return out


def inspect(client):
    for b in recent_batches(client, 30):
        print(f"{b.id}  {b.created_at}  {b.processing_status:12s} "
              f"total={batch_total(b)}")


def known_ids(all_states):
    ids = set()
    for st in all_states.values():
        for arm_ids in st.values():
            ids.update(arm_ids)
    return ids


def adopt_and_submit(client):
    """Ensure every chunk of every arm has a batch id; adopt phantoms."""
    submit_client = anthropic.Anthropic(max_retries=0)
    states = {s: load_state(d) for s, d in STUDIES.items()}
    plans = []   # (study, arm, chunk_lists)
    for study, out_dir in STUDIES.items():
        for arm in ARMS:
            reqs = C.load_jsonl(req_path(out_dir, arm))
            plans.append((study, arm, reqs, chunks_of(reqs)))

    listed = recent_batches(client, 100)
    used = known_ids(states)
    # candidate phantoms: not canceled, not already recorded
    cands = [b for b in listed
             if b.id not in used
             and b.processing_status != "canceling"
             and not (b.request_counts.canceled == batch_total(b) > 0)]

    def adopt_by_size(n, label):
        """Adopt the newest unrecorded batch with exactly n requests;
        cancel older duplicates of the same size."""
        matches = [b for b in cands if batch_total(b) == n]
        if not matches:
            return None
        matches.sort(key=lambda b: str(b.created_at), reverse=True)
        keep = matches[0]
        for extra in matches[1:]:
            print(f"  cancelling duplicate phantom {extra.id} (n={n})")
            try:
                client.messages.batches.cancel(extra.id)
            except anthropic.APIError as e:
                print(f"  cancel failed ({e}); leaving it")
        cands.remove(keep)
        print(f"  ADOPTED phantom {keep.id} as {label} (n={n})")
        return keep.id

    for study, arm, reqs, chks in plans:
        st = states[study]
        out_dir = STUDIES[study]
        # 1) whole-arm phantom from the failed single-shot submission?
        if not st[arm]:
            bid = adopt_by_size(len(reqs), f"{study}/{arm} (whole arm)")
            if bid:
                st[arm] = [bid]
                save_state(out_dir, st)
                continue
        if len(st[arm]) == 1 and len(chks) > 1:
            continue  # whole-arm batch already recorded
        # 2) per-chunk submission with phantom re-identification
        for i, chunk in enumerate(chks):
            if i < len(st[arm]):
                continue
            bid = adopt_by_size(len(chunk), f"{study}/{arm} chunk {i}")
            if bid is None:
                print(f"submitting {study}/{arm} chunk {i} "
                      f"({len(chunk)} requests)...")
                batch = submit_client.messages.batches.create(requests=[
                    {"custom_id": r["custom_id"], "params": r["params"]}
                    for r in chunk])
                bid = batch.id
                print(f"  -> {bid}")
            st[arm].append(bid)
            save_state(out_dir, st)
    return states


def poll_all(client, states):
    pending = {(s, a, bid)
               for s, st in states.items() for a in ARMS for bid in st[a]}
    while pending:
        done = set()
        for s, a, bid in sorted(pending):
            b = client.messages.batches.retrieve(bid)
            c = b.request_counts
            print(f"[{time.strftime('%H:%M:%S')}] {s}/{a} {bid}: "
                  f"{b.processing_status} (proc={c.processing} "
                  f"ok={c.succeeded} err={c.errored})", flush=True)
            if b.processing_status == "ended":
                done.add((s, a, bid))
        pending -= done
        if pending:
            time.sleep(POLL_SECONDS)


def _extract(message):
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


def fetch_all(client, states):
    for study, out_dir in STUDIES.items():
        refusals_path = os.path.join(out_dir, "ext_refusals.jsonl")

        def log_refusal(rec):
            with open(refusals_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

        for arm in ARMS:
            requests_by_id = {r["custom_id"]: r["params"]
                              for r in C.load_jsonl(req_path(out_dir, arm))}
            results, retry_ids = {}, []
            for bid in states[study][arm]:
                for result in client.messages.batches.results(bid):
                    cid = result.custom_id
                    rtype = result.result.type
                    if rtype == "succeeded":
                        msg = result.result.message
                        stop_reason, parsed, text = _extract(msg)
                        if stop_reason == "refusal":
                            log_refusal({"custom_id": cid, "source": "batch",
                                         "arm": arm, "raw_text": text})
                            retry_ids.append(cid)
                        elif parsed is None:
                            results[cid] = {
                                "custom_id": cid, "status": "missing",
                                "stop_reason": stop_reason,
                                "note": "unparseable", "raw_text": text}
                        else:
                            results[cid] = {
                                "custom_id": cid, "status": "ok",
                                "stop_reason": stop_reason, "parsed": parsed}
                    elif rtype in ("errored", "expired"):
                        err = getattr(result.result, "error", None)
                        results[cid] = {"custom_id": cid, "status": "missing",
                                        "note": rtype,
                                        "error": str(err) if err else None}
                        retry_ids.append(cid)
                    else:
                        results[cid] = {"custom_id": cid, "status": "missing",
                                        "note": rtype}
            for cid in retry_ids:
                params = requests_by_id.get(cid)
                if params is None:
                    continue
                try:
                    msg = client.messages.create(**params)
                except anthropic.APIError as e:
                    results[cid] = {"custom_id": cid, "status": "missing",
                                    "note": "live_retry_failed",
                                    "error": str(e)}
                    continue
                stop_reason, parsed, text = _extract(msg)
                if stop_reason == "refusal" or parsed is None:
                    if stop_reason == "refusal":
                        log_refusal({"custom_id": cid, "source": "live_retry",
                                     "arm": arm, "raw_text": text})
                    results[cid] = {
                        "custom_id": cid, "status": "missing",
                        "stop_reason": stop_reason,
                        "note": "refused_or_unparseable_after_retry"}
                else:
                    results[cid] = {"custom_id": cid, "status": "ok",
                                    "stop_reason": stop_reason,
                                    "parsed": parsed, "note": "live_retry"}
            for cid in requests_by_id:
                if cid not in results:
                    results[cid] = {"custom_id": cid, "status": "missing",
                                    "note": "absent_from_batch_results"}
            out_path = os.path.join(out_dir, f"ext_{arm}_results.jsonl")
            with open(out_path, "w") as fh:
                for cid in sorted(results):
                    fh.write(json.dumps(results[cid]) + "\n")
            n_ok = sum(1 for r in results.values() if r["status"] == "ok")
            print(f"{study}/{arm}: {n_ok}/{len(requests_by_id)} ok "
                  f"-> {out_path}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    client = anthropic.Anthropic()
    if cmd == "inspect":
        inspect(client)
        return
    if cmd != "run":
        raise SystemExit(f"unknown command {cmd!r} (run|inspect)")
    states = adopt_and_submit(client)
    total = sum(len(st[a]) for st in states.values() for a in ARMS)
    print(f"all arms covered by {total} batches; polling...")
    poll_all(client, states)
    fetch_all(client, states)
    print("EXT_BATCHES_DONE")


if __name__ == "__main__":
    main()
