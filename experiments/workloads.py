"""Generated deterministic workloads and narrowly scored held-out answers."""
import base64
import struct
import zlib

QUALITY = [
 ('What does Python expression `len({1, 1, 2, 3})` evaluate to? Reply only with the integer.', '3'),
 ('A job starts with 11 queued tasks, completes 4, and adds 6. How many remain? Reply only with the integer.', '13'),
 ('For `values = [4, 7, 9]`, what is `values[-2]` in Python? Reply only with the integer.', '7'),
 ('A function returns true only if both inputs are true. What Boolean operation is this? Reply only with its uppercase name.', 'AND'),
 ('A test suite has 5 files and each has 8 tests. How many tests are there? Reply only with the integer.', '40'),
]

def image_url():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    # 512x512 synthetic image: red field with a large contrasting blue square.
    rows = []
    for y in range(512):
        row = bytearray()
        for x in range(512):
            row.extend((0, 0, 255) if 160 <= x < 352 and 160 <= y < 352 else (255, 0, 0))
        rows.append(b'\0' + row)
    raw = b''.join(rows)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 512, 512, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def make_workload(name, nonce='', index=0, tokenizer=None, answer_policy='lookup'):
    if name in ('512', '8k', '32k', '48k', '64k'):
        target = {'512': 512, '8k': 8192, '32k': 32768, '48k': 49152, '64k': 65536}[name]
        head = (nonce + '\n' if nonce else '') + 'Review these generated repository excerpts.\n'
        tail = '\nQuestion: What is the sentinel value in config? Reply with only the number.\nconfig: sentinel=37'
        if answer_policy == 'code':
            tail = ('\nImplement a Python function enabled_batches(records, limit) that yields batches of enabled records '
                    'in input order. Reject a nonpositive limit and include a final partial batch. '
                    'Provide the function and two short usage examples.')
        def block(i):
            return (f'\n# file: services/component_{i:04d}.py\n'
                    f'def process_{i:04d}(records, retries={i % 7 + 1}):\n'
                    f'    """Validate batch {i:04d} and preserve insertion order."""\n'
                    f'    result = [item for item in records if item.get("enabled", False)]\n'
                    f'    return {{"count": len(result), "retries": retries, "items": result}}\n'
                    f'# test_{i:04d}: empty input returns count zero; invalid records are excluded.\n')
        blocks=[]
        # Tokenizer calibration applies to message content, not chat-template special tokens.
        size = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids) if tokenizer else len(s) // 4
        # Estimate from one sample, then extend in larger increments to avoid
        # re-tokenizing every preceding file each time another file is added.
        approx = max(1, size(block(0)))
        remaining = target - size(head + tail)
        while remaining > 0:
            begin = len(blocks)
            blocks.extend(block(i) for i in range(begin, begin + max(1, remaining // approx)))
            remaining = target - size(head + ''.join(blocks) + tail)
        text=head + ''.join(blocks) + tail
        if tokenizer:
            # Trim context only, retaining question and suffix; token boundaries can differ at joins.
            for _ in range(128):
                if not blocks or size(text) <= target:
                    break
                previous = text
                last=blocks[-1]
                excess=size(text)-target
                ids=tokenizer.encode(last, add_special_tokens=False).ids
                if len(ids) <= excess:
                    blocks.pop()
                else:
                    blocks[-1]=tokenizer.decode(ids[:len(ids)-excess])
                text=head + ''.join(blocks) + tail
                if blocks and not blocks[-1]: blocks.pop()
                if text == previous:
                    break
            # Report actual counts separately; no guarantee of exactness across token boundaries.
        return [{'role': 'user', 'content': text}], '37' if answer_policy == 'lookup' else None
    if name == 'quality':
        prompt, answer = QUALITY[index % len(QUALITY)]
        return [{'role':'user','content':prompt}], answer
    if name == 'tool':
        return [{'role':'user','content':'Use the calculator tool to calculate 3*7, then reply only with the result.'}], '21'
    if name == 'image':
        return [{'role':'user','content':[{'type':'text','text':'What is the dominant background color? Reply with one word.'},{'type':'image_url','image_url':{'url':image_url()}}]}], 'red'
    raise ValueError(name)
