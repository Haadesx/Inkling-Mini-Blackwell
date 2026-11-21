#!/usr/bin/env python3
"""Run a deterministic long-prefix, six-tool agent session against a live local API."""

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from workloads import make_workload

RECORDS = [
    ("alder", "raven", 14, "birch"),
    ("birch", "cedar", 27, "cobalt"),
    ("cobalt", "heron", 9, "dune"),
    ("dune", "lumen", 43, "ember"),
    ("ember", "maple", 32, "fjord"),
    ("fjord", "quartz", 21, None),
]
LONG_RECORDS = [
    (name, f"code_{i:02d}", (i * 37 + 11) % 101,
     names_next)
    for i, (name, names_next) in enumerate(zip(
        ('alder', 'birch', 'cobalt', 'dune', 'ember', 'fjord', 'grove', 'harbor',
         'indigo', 'juniper', 'kestrel', 'lantern', 'meadow', 'north', 'onyx', 'pine'),
        ('birch', 'cobalt', 'dune', 'ember', 'fjord', 'grove', 'harbor', 'indigo',
         'juniper', 'kestrel', 'lantern', 'meadow', 'north', 'onyx', 'pine', None),
    ))
]
TOOL = {
    "type": "function",
    "function": {
        "name": "read_release_record",
        "description": "Read one release record by exact name. The returned next_name identifies the next record; null means the chain ends.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_stream(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start = time.perf_counter()
    first = None
    first_final = None
    content = []
    reasoning = []
    calls = {}
    usage = None
    finish = None
    details = []
    timings = []
    sse = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        lines = []

        def process(raw):
            nonlocal first, first_final, usage, finish
            if raw == "[DONE]":
                return
            obj = json.loads(raw)
            if obj.get("error"):
                raise ValueError(str(obj["error"]))
            if obj.get("usage") is not None:
                usage = obj["usage"]
            if obj.get("sglext") is not None:
                details.append(obj["sglext"])
            if obj.get("timings") is not None:
                timings.append(obj["timings"])
            for choice in obj.get("choices") or []:
                if choice.get("timings") is not None:
                    timings.append(choice["timings"])
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text or thought or delta.get("tool_calls"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    if text and first_final is None:
                        first_final = now
                if text:
                    content.append(text)
                if thought:
                    reasoning.append(thought)
                for part in delta.get("tool_calls") or []:
                    item = calls.setdefault(
                        part.get("index", 0),
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    item["id"] += part.get("id") or ""
                    for key in ("name", "arguments"):
                        item["function"][key] += (part.get("function") or {}).get(
                            key
                        ) or ""
            # Preserve timing/speculation frames without duplicating streamed text.
            if obj.get("timings") is not None or obj.get("sglext") is not None:
                sse.append(obj)

        for line in response:
            line = line.decode("utf-8").rstrip("\r\n")
            if not line:
                if lines:
                    process("\n".join(lines))
                    lines = []
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip())
        if lines:
            process("\n".join(lines))
    end = time.perf_counter()
    return {
        "wall_s": end - start,
        "ttft_s": first - start if first is not None else None,
        "first_content_s": first_final - start if first_final is not None else None,
        "answer": "".join(content),
        "reasoning": "".join(reasoning),
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
        "usage": usage,
        "server_timings": timings,
        "sglext": details,
        "timing_spec_frames": sse,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    p.add_argument("--engine", choices=["llama", "basalt", "sglang"], required=True)
    p.add_argument("--model", default="inkling-mini")
    p.add_argument(
        "--tokenizer", type=pathlib.Path, required=True, help="Native tokenizer.json"
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="New JSON output (not overwritten)",
    )
    p.add_argument("--size", choices=["8k", "32k", "64k"], default="32k")
    p.add_argument("--chain-length", type=int, choices=[6, 16], default=6,
                   help="Number of sequential tool calls (default: 6; 16 is the long fixture)")
    p.add_argument("--run-id", default="agent-session-v1")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="For llama, set cache_prompt=false on every request",
    )
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and check prefix, without contacting server",
    )
    a = p.parse_args()
    if a.max_tokens < 1 or a.timeout <= 0:
        p.error("max-tokens and timeout must be positive")
    if not a.dry_run and a.output.exists():
        p.error("output already exists")
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(a.tokenizer))
    except Exception as exc:
        p.error("cannot load tokenizer: " + str(exc))
    prefix = make_workload(a.size, "Run marker: " + a.run_id, tokenizer=tok)[0][0][
        "content"
    ]
    token_count = len(tok.encode(prefix, add_special_tokens=False).ids)
    target = {"8k": 8192, "32k": 32768, "64k": 65536}[a.size]
    if token_count != target:
        p.error(
            f"prefix calibration returned {token_count} content tokens, expected {target}"
        )
    instruction = (
        "The preceding repository excerpt includes an unrelated example question. Ignore that example question. "
        "Follow the release-record chain with the read_release_record tool. Start with name alder. "
        "Call the tool exactly once for each record, following each returned next_name until it is null. "
        "After reading all six records, identify the record with the largest value and return ONLY JSON "
        'with keys "name", "code", and "value". The records and their values are available only from the tool.'
    )
    records = RECORDS
    if a.chain_length == 16:
        records = LONG_RECORDS
        instruction = instruction.replace("all six records", "all sixteen records")
    messages = [{"role": "user", "content": prefix + "\n\n" + instruction}]
    result = {
        "version": 1,
        "engine": a.engine,
        "model": a.model,
        "run_id": a.run_id,
        "size": a.size,
        "prefix_content_tokens": token_count,
        "prefix_sha256": digest(prefix),
        "user_message_sha256": digest(messages[0]["content"]),
        "tokenizer_sha256": hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "max_tokens": a.max_tokens,
        "seed": a.seed,
        "reasoning_effort": a.reasoning_effort,
        "tool_schema": TOOL,
        "cache_policy": ("off" if a.no_cache else "first off, followups on")
        if a.engine == "llama"
        else "engine default",
        "turns": [],
        "status": "incomplete",
        "error": None,
    }
    if a.chain_length != 6:
        result["chain_length"] = a.chain_length
    if a.dry_run:
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "prefix_content_tokens",
                        "prefix_sha256",
                        "user_message_sha256",
                        "tokenizer_sha256",
                    )
                }
            )
        )
        return 0
    started = time.perf_counter()
    try:
        for turn in range(len(records) + 1):
            body = {
                "model": a.model,
                "messages": messages,
                "tools": [TOOL],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": a.max_tokens,
                "seed": a.seed,
                "stream": True,
                "stream_options": {"include_usage": True},
                "reasoning_effort": a.reasoning_effort,
            }
            if a.engine == "llama":
                body["cache_prompt"] = bool(turn and not a.no_cache)
            if a.engine == "sglang":
                body["return_spec_tokens_details"] = True
            item = parse_stream(
                a.base_url.rstrip("/") + "/chat/completions", body, a.timeout
            )
            item["turn"] = turn + 1
            item["cache_prompt_requested"] = body.get("cache_prompt")
            usage = item.get("usage") or {}
            cache_details = usage.get("prompt_tokens_details") or {}
            cached = (
                cache_details.get("cached_tokens")
                if isinstance(cache_details, dict)
                else None
            )
            prompt_tokens = usage.get("prompt_tokens")
            item["prompt_tokens"] = prompt_tokens
            item["cached_prompt_tokens"] = cached
            item["new_prompt_tokens"] = (
                prompt_tokens - cached
                if isinstance(prompt_tokens, int) and isinstance(cached, int)
                else None
            )
            result["turns"].append(item)
            calls = item["tool_calls"]
            if turn < len(records):
                if (
                    len(calls) != 1
                    or not calls[0]["id"]
                    or calls[0]["function"]["name"] != "read_release_record"
                ):
                    raise ValueError(
                        f"turn {turn + 1}: expected exactly one read_release_record call with an ID"
                    )
                try:
                    args = json.loads(calls[0]["function"]["arguments"])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"turn {turn + 1}: invalid tool arguments"
                    ) from None
                name, code, value, next_name = records[turn]
                if args != {"name": name}:
                    raise ValueError(
                        f"turn {turn + 1}: unexpected tool arguments: {args!r}"
                    )
                tool_output = {
                    "name": name,
                    "code": code,
                    "value": value,
                    "next_name": next_name,
                    "payload": f"Release record {name} verified.",
                }
                item["tool_result"] = tool_output
                assistant = {
                    "role": "assistant",
                    "content": item["answer"] or None,
                    "tool_calls": calls,
                }
                if item["reasoning"]:
                    assistant["reasoning_content"] = item["reasoning"]
                messages.extend(
                    [
                        assistant,
                        {
                            "role": "tool",
                            "tool_call_id": calls[0]["id"],
                            "content": json.dumps(tool_output, separators=(",", ":")),
                        },
                    ]
                )
            else:
                if calls:
                    raise ValueError(f"turn {turn + 1}: unexpected extra tool call")
                try:
                    answer = json.loads(item["answer"].strip())
                except ValueError:
                    raise ValueError(f"turn {turn + 1}: final answer is not JSON") from None
                best = max(records, key=lambda record: record[2])
                expected = {"name": best[0], "code": best[1], "value": best[2]}
                if answer != expected:
                    raise ValueError(f"turn {turn + 1}: incorrect final answer: {answer!r}")
                result["status"] = "passed"
    except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
        result["error"] = str(exc)
        result["status"] = "failed"
    result["session_wall_s"] = time.perf_counter() - started
    result["first_request_wall_s"] = (
        result["turns"][0]["wall_s"] if result["turns"] else None
    )
    result["followup_wall_s"] = (
        sum(t["wall_s"] for t in result["turns"][1:])
        if len(result["turns"]) > 1
        else None
    )
    for key in ("prompt_tokens", "completion_tokens"):
        values = [(t.get("usage") or {}).get(key) for t in result["turns"]]
        result["total_" + key] = (
            sum(values) if values and all(isinstance(v, int) for v in values) else None
        )
    try:
        with a.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError as exc:
        p.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(a.output),
                "status": result["status"],
                "error": result["error"],
                "turns": len(result["turns"]),
            }
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
