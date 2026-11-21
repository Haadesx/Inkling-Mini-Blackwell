#!/usr/bin/env python3
"""Score visible completed-code answers with a restricted AST and timed subprocess."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Constant,
    ast.If, ast.For, ast.Assign, ast.AugAssign, ast.Return, ast.Raise, ast.Yield,
    ast.Name, ast.Load, ast.Store, ast.List, ast.Tuple, ast.Dict, ast.Subscript,
    ast.Slice, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Call,
    ast.Attribute, ast.Pass, ast.Break, ast.Continue, ast.keyword,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn, ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mod,
}
SAFE_CALLS = {'len', 'ValueError', 'TypeError', 'range', 'enumerate', 'isinstance'}
SAFE_METHODS = {'get', 'append'}
ANNOTATION_ATOMS = {'Any', 'int', 'str', 'bool', 'float', 'bytes'}
ANNOTATION_CONTAINERS = {
    'List': 1, 'list': 1, 'Iterable': 1, 'Iterator': 1,
    'Dict': 2, 'dict': 2,
}


def validate_annotation(annotation):
    """Accept only bounded type expressions; never evaluate them as code."""
    count = 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if depth > 8 or count > 64:
            raise ValueError('annotation too complex')
        if isinstance(node, ast.Name) and node.id in ANNOTATION_ATOMS:
            return
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            arity = ANNOTATION_CONTAINERS.get(node.value.id)
            if arity is not None:
                items = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
                if len(items) == arity:
                    for item in items:
                        visit(item, depth + 1)
                    return
        raise ValueError('unsupported annotation')

    visit(annotation, 0)


class StripAnnotations(ast.NodeTransformer):
    """Annotations have already been checked and must not execute at runtime."""

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        return ast.copy_location(ast.Assign(targets=[node.target], value=self.visit(node.value)), node)


def extract(answer):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', answer, re.S | re.I)
    candidates = blocks + [answer]
    found = []
    for text in candidates:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        found.extend(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'enabled_batches')
        if found:
            break
    if len(found) != 1:
        raise ValueError('missing, incomplete, or ambiguous function')
    fn = found[0]
    if fn.decorator_list or fn.args.defaults or fn.args.kw_defaults:
        raise ValueError('decorators or defaults unsupported')
    if len(fn.args.args) != 2 or [a.arg for a in fn.args.args] != ['records', 'limit']:
        raise ValueError('unsupported function signature')
    if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        raise ValueError('unsupported function signature')
    # Validate and remove annotations before the ordinary executable AST checks.
    for node in ast.walk(fn):
        if isinstance(node, ast.arg) and node.annotation is not None:
            validate_annotation(node.annotation)
        elif isinstance(node, ast.FunctionDef) and node.returns is not None:
            validate_annotation(node.returns)
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.simple != 1 or node.value is None:
                raise ValueError('unsupported annotated assignment')
            validate_annotation(node.annotation)
    fn = StripAnnotations().visit(fn)
    ast.fix_missing_locations(fn)
    # Permit only the observed local generator pattern: a single direct,
    # undecorated zero-argument ``gen`` followed by ``return gen()``.
    # No other reference to its name, recursive call or nested function is allowed.
    nested = [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn]
    generator_call = None
    if nested:
        if len(nested) != 1 or nested[0] not in fn.body:
            raise ValueError('unsupported nested function')
        gen = nested[0]
        args = gen.args
        if (gen.name != 'gen' or gen.decorator_list or args.posonlyargs or args.args
                or args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.kw_defaults
                or not any(isinstance(n, ast.Yield) for n in ast.walk(gen))):
            raise ValueError('unsupported local generator')
        last = fn.body[-1]
        if (not isinstance(last, ast.Return) or not isinstance(last.value, ast.Call)
                or not isinstance(last.value.func, ast.Name) or last.value.func.id != 'gen'
                or last.value.args or last.value.keywords):
            raise ValueError('unsupported local generator return')
        refs = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'gen']
        if len(refs) != 1 or refs[0] is not last.value.func:
            raise ValueError('unsupported local generator reference')
        generator_call = last.value
    for node in ast.walk(fn):
        if type(node) not in ALLOWED:
            raise ValueError(f'unsupported syntax: {type(node).__name__}')
        if isinstance(node, ast.Name) and '__' in node.id:
            raise ValueError('dunder name')
        if isinstance(node, ast.Attribute) and (node.attr not in SAFE_METHODS or '__' in node.attr):
            raise ValueError('unsupported attribute')
        if isinstance(node, ast.Call):
            if node is generator_call:
                continue
            if isinstance(node.func, ast.Name) and node.func.id in SAFE_CALLS:
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in SAFE_METHODS:
                continue
            raise ValueError('unsupported call')
        if isinstance(node, ast.For) and not (isinstance(node.iter, ast.Name) and node.iter.id == 'records'):
            raise ValueError('unsupported loop iterable')
    return ast.unparse(fn)


WORKER = r'''
import json, sys
source = sys.stdin.read()
namespace = {'__builtins__': {'len': len, 'ValueError': ValueError, 'TypeError': TypeError, 'range': range, 'enumerate': enumerate, 'isinstance': isinstance,
                               'int': int, 'bool': bool, 'dict': dict}}
exec(compile(source, '<answer>', 'exec'), namespace)
f = namespace['enabled_batches']
a = [{'id': 0, 'enabled': True}, {'id': 1, 'enabled': False}, {'id': 2, 'enabled': True}, {'id': 3, 'enabled': True}, {'id': 4}, {'id': 5, 'enabled': True}, {'id': 6, 'enabled': True}]
cases = {
  'partial_order_disabled': lambda: list(f(a, 2)) == [[a[0], a[2]], [a[3], a[5]], [a[6]]],
  'all_disabled': lambda: list(f([{'enabled': False}, {}], 2)) == [],
  'empty': lambda: list(f([], 2)) == [],
  'large_limit': lambda: list(f(a, 9)) == [[a[0], a[2], a[3], a[5], a[6]]],
  'unit_limit': lambda: list(f(a, 1)) == [[a[0]], [a[2]], [a[3]], [a[5]], [a[6]]],
}
def invalid(limit):
    try: list(f(a, limit))
    except ValueError: return True
    return False
cases['zero_limit'] = lambda: invalid(0)
cases['negative_limit'] = lambda: invalid(-1)
results = {}
for name, test in cases.items():
    try: results[name] = bool(test())
    except Exception: results[name] = False
print(json.dumps(results))
'''


def score(path):
    result = {'file': str(path), 'status': 'missing', 'passed': None}
    if not path.exists():
        return result
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) != 1:
            raise ValueError('expected exactly one response')
        record = records[0]
        result.update(ok=record.get('ok'), finish_reason=record.get('finish_reason'),
                      max_tokens=record.get('max_tokens'))
        answer = record.get('answer') or ''
        result['answer_sha256'] = hashlib.sha256(answer.encode()).hexdigest()
        if not answer.strip():
            result['status'] = 'missing_answer'
            return result
        if record.get('max_tokens') not in (1024, 3072) or not record.get('ok') or record.get('finish_reason') != 'stop':
            result['status'] = 'incomplete_or_wrong_budget'
            return result
        code = extract(answer)
        proc = subprocess.run([sys.executable, '-I', '-c', WORKER], input=code, text=True,
                              capture_output=True, timeout=3, check=True)
        tests = json.loads(proc.stdout)
        result.update(status='scored', tests=tests, passed=all(tests.values()))
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        result.update(status='unsupported_or_error', error=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('directory does not exist')
    results = {p.name: score(p / 'completed-code.jsonl')
               for p in sorted(args.directory.iterdir()) if p.is_dir()}
    args.output.write_text(json.dumps({'scope': 'visible completed-code answers at 1024 and separately labeled 3072-token budgets',
                                      'arms': results}, indent=2) + '\n')
    for arm, result in results.items():
        print(f"{arm}: {result['status']} passed={result['passed']}")


if __name__ == '__main__':
    main()
