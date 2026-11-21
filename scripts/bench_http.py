#!/usr/bin/env python3
"""Run streaming OpenAI-compatible chat requests; writes one JSONL record per request."""
import argparse
import concurrent.futures
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'experiments'))
from workloads import make_workload


def request_one(args, index, messages=None, tools=None):
    nonce = '' if args.prefix == 'shared' else 'Run marker: ' + args.run_id + '-' + str(index)
    generated, expected = args.prepared[index]
    messages = generated if messages is None else messages
    # Include the marker for non-length workloads too.
    if nonce and args.workload not in ('512', '8k', '32k', '48k', '64k') and messages is generated:
        messages.insert(0, {'role': 'system', 'content': nonce})
    payload = {'model': args.model, 'messages': messages, 'stream': True, 'stream_options': {'include_usage': True}, 'temperature': 0, 'max_tokens': args.max_tokens}
    payload['reasoning_effort'] = args.reasoning_effort
    if args.engine == 'llama':
        payload['cache_prompt'] = args.workload == 'tool' and args.tool_roundtrip
    if args.engine == 'sglang':
        payload['return_spec_tokens_details'] = True
    if tools is not None:
        payload['tools'] = tools
        payload['tool_choice'] = 'required'
    if args.seed is not None:
        payload['seed'] = args.seed
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            return {'index': index, 'error': 'API key environment variable is missing', 'ok': False}
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    first = None
    first_content = None
    calls = {}
    events = []
    raw_events = []
    pieces = []
    reasoning = []
    usage = None
    finish = None
    errors = []
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            data_lines = []
            def process(data):
                nonlocal first, first_content, usage, finish
                raw_events.append({'elapsed_s': time.perf_counter() - start, 'data': data})
                if data == '[DONE]':
                    return
                obj = json.loads(data)
                if obj.get('error'):
                    errors.append(obj['error'])
                if obj.get('usage') is not None:
                    usage = obj['usage']
                for choice in obj.get('choices') or []:
                    if choice.get('finish_reason') is not None:
                        finish = choice['finish_reason']
                    delta = choice.get('delta') or {}
                    content = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if isinstance(content, str) and content:
                        pieces.append(content)
                        if first_content is None:
                            first_content = time.perf_counter()
                    for call in delta.get('tool_calls') or []:
                        item = calls.setdefault(call.get('index', 0), {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if call.get('id'): item['id'] += call['id']
                        fn = call.get('function') or {}
                        for field in ('name', 'arguments'):
                            item['function'][field] += fn.get(field) or ''
                    if isinstance(thought, str) and thought:
                        reasoning.append(thought)
                    if content or thought or delta.get('tool_calls'):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        events.append(now)
            for raw in response:
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    if data_lines:
                        process('\n'.join(data_lines))
                        data_lines = []
                elif line.startswith('data:'):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                process('\n'.join(data_lines))
    except urllib.error.HTTPError as exc:
        errors.append('HTTP %s: %s' % (exc.code, exc.read(2000).decode('utf-8', 'replace')))
    except (OSError, ValueError, TypeError) as exc:
        errors.append('%s: %s' % (type(exc).__name__, exc))
    end = time.perf_counter()
    answer = ''.join(pieces)
    gaps = [b - a for a, b in zip(events, events[1:])]
    completion = usage.get('completion_tokens') if isinstance(usage, dict) else None
    # Each SSE event can contain multiple tokens; gap is event-level, not token ITL.
    decode_s = end - first if first is not None else None
    decode_rate = ((completion - 1) / (events[-1] - first) if isinstance(completion, int) and completion > 1 and len(events) > 1 and events[-1] > first else None)
    normalized = answer.strip().lower().strip(' .!\n\t')
    content_tokens = None
    if args.tokenizer and len(messages) == 1 and isinstance(messages[0].get('content'), str):
        content_tokens = len(args.tokenizer.encode(messages[0]['content'], add_special_tokens=False).ids)
    return {'index': index, 'run_id': args.run_id, 'model': args.model, 'concurrency': args.concurrency,
            'seed': args.seed, 'reasoning_effort': args.reasoning_effort, 'max_tokens': args.max_tokens, 'ok': not errors and finish is not None,
            'errors': errors, 'workload': args.workload, 'prefix': args.prefix,
            'wall_s': end - start, 'ttft_s': first - start if first is not None else None,
            'first_final_content_s': first_content - start if first_content is not None else None,
            'tool_calls': [calls[k] for k in sorted(calls)],
            'first_to_end_s': decode_s, 'content_event_count': len(events), 'event_gap_mean_s': statistics.mean(gaps) if gaps else None,
            'event_gaps_s': gaps, 'token_itl_s': None, 'approx_decode_tokens_per_s': decode_rate,
            'decode_rate_note': 'Approximation: (completion_tokens - 1) / first-to-last content event; assumes first event is one token. Completion may include reasoning.' if decode_rate is not None else 'Unavailable: needs usage and at least two content events.',
            'usage': usage, 'finish_reason': finish, 'answer': answer, 'reasoning': ''.join(reasoning),
            'raw_sse': raw_events, 'expected': expected,
            'exact_match': normalized == expected.lower() if expected is not None and answer and not errors and finish is not None else None,
            'calibrated_content_tokens': content_tokens,
            'request_messages': messages if args.include_prompts else None}


def tool_roundtrip(args, index):
    nonce = '' if args.prefix == 'shared' else 'Run marker: ' + args.run_id + '-' + str(index)
    messages, _ = make_workload('tool', nonce, index, args.tokenizer)
    if nonce:
        messages.insert(0, {'role': 'system', 'content': nonce})
    schema = [{'type': 'function', 'function': {'name': 'calculator', 'description': 'Calculate 3*7', 'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression'], 'additionalProperties': False}}}]
    start = time.perf_counter()
    first = request_one(args, index, messages, schema)
    calls = first['tool_calls']
    valid = False
    if len(calls) == 1 and calls[0]['id'] and calls[0]['function']['name'] == 'calculator':
        try:
            arguments = json.loads(calls[0]['function']['arguments'])
            valid = arguments == {'expression': '3*7'}
        except (ValueError, TypeError):
            pass
    if not valid:
        return {'index': index, 'ok': False, 'error': 'invalid tool call', 'tool_valid': False, 'first_response': first, 'second_response': None, 'wall_s': time.perf_counter()-start, 'usage': None}
    messages = messages + [{'role': 'assistant', 'content': first['answer'] or None, 'tool_calls': calls}, {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '21'}]
    # The tool result is fixed from the validated expression; no model-generated code is executed.
    second = request_one(args, index, messages)
    usage = None
    if isinstance(first.get('usage'), dict) and isinstance(second.get('usage'), dict):
        a, b = first['usage'].get('completion_tokens'), second['usage'].get('completion_tokens')
        if isinstance(a, int) and isinstance(b, int): usage = {'completion_tokens': a+b}
    return {'index': index, 'ok': first['ok'] and second['ok'] and second['exact_match'] is True,
            'tool_valid': True, 'exact_match': second['exact_match'], 'first_response': first, 'second_response': second,
            'first_final_content_s': first['wall_s'] + second['first_final_content_s'] if second['first_final_content_s'] is not None else None,
            'wall_s': time.perf_counter()-start, 'usage': usage}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1', help='OpenAI API base ending in /v1')
    p.add_argument('--model', required=True)
    p.add_argument('--reasoning-effort', choices=['none', 'minimal', 'low', 'medium', 'high'], default='none')
    p.add_argument('--answer-policy', choices=['lookup', 'code'], default='lookup')
    p.add_argument('--engine', choices=['llama', 'sglang', 'basalt'], default='llama')
    p.add_argument('--workload', choices=['512', '8k', '32k', '48k', '64k', 'tool', 'quality', 'image'], default='512')
    p.add_argument('--concurrency', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--requests', type=int, default=1, help='Total requests, not requests per worker')
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run-id', help='Explicit ID makes unique-prefix prompts reproducible across paired runs')
    p.add_argument('--tokenizer-json', help='Optional exact native tokenizer.json for content-token calibration (requires tokenizers package)')
    p.add_argument('--tool-roundtrip', action='store_true', help='Execute a validated calculator tool round trip')
    p.add_argument('--prefix', choices=['shared', 'unique'], default='unique', help='Shared repeats the prompt; unique prepends a request marker. Does not clear server caches.')
    p.add_argument('--api-key-env', help='Read bearer token from this environment variable; never log it')
    p.add_argument('--include-prompts', action='store_true', help='Include generated prompts in JSONL')
    p.add_argument('--output', default='-', help='JSONL output file, or - for stdout')
    args = p.parse_args()
    if args.requests < 1 or args.max_tokens < 1 or args.timeout <= 0:
        p.error('requests, max-tokens, and timeout must be positive')
    args.run_id = args.run_id or uuid.uuid4().hex
    args.tokenizer = None
    if args.tokenizer_json:
        try:
            from tokenizers import Tokenizer
            args.tokenizer = Tokenizer.from_file(args.tokenizer_json)
        except Exception as exc:
            p.error('cannot load tokenizer: ' + str(exc))
    if args.tool_roundtrip and args.workload != 'tool':
        p.error('--tool-roundtrip requires --workload tool')
    args.prepared = []
    for index in range(args.requests):
        nonce = '' if args.prefix == 'shared' else f'Run marker: {args.run_id}-{index}'
        args.prepared.append(make_workload(args.workload, nonce, index, args.tokenizer, args.answer_policy))
    batch_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: tool_roundtrip(args, i) if args.tool_roundtrip else request_one(args, i), range(args.requests)))
    batch_span = time.perf_counter() - batch_start
    counts = [r.get('usage', {}).get('completion_tokens') if isinstance(r.get('usage'), dict) else None for r in results]
    aggregate = sum(counts) / batch_span if all(isinstance(n, int) for n in counts) else None
    for r in results:
        r['batch_span_s'] = batch_span
        r['batch_completion_tokens_per_s'] = aggregate
    output = sys.stdout if args.output == '-' else open(args.output, 'w', encoding='utf-8')
    try:
        for result in results:
            print(json.dumps(result, ensure_ascii=False), file=output)
    finally:
        if output is not sys.stdout:
            output.close()
    return 0 if all(r['ok'] for r in results) else 1

if __name__ == '__main__':
    sys.exit(main())
