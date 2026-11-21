#!/usr/bin/env python3
"""One small streaming correctness check against an OpenAI-compatible server."""
import sys
from bench_http import main

if __name__ == '__main__':
    if '--workload' not in sys.argv:
        sys.argv += ['--workload', 'tool', '--tool-roundtrip']
    if '--max-tokens' not in sys.argv:
        sys.argv += ['--max-tokens', '32']
    sys.exit(main())
