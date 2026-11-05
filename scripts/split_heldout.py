"""Write copies of run files without the 25 dev questions (held-out 175).

Dev ids are the question_ids of runs/agent_dev25.jsonl. Lines are copied
verbatim so extra fields survive.

Run from the repo root:
    venv/bin/python -m scripts.split_heldout OUT_DIR RUN_FILE [RUN_FILE ...]
    e.g. venv/bin/python -m scripts.split_heldout runs/heldout175 \
        runs/agent_200.jsonl ../rag-eval-harness/runs/dense_rerank_200.jsonl
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .common import (
    check_unique_names,
    dev_ids,
    drop_dev,
    read_lines,
    write_lines,
)


def run(argv: Sequence[str] | None) -> int:
    """Filter each run file into OUT_DIR under the same name."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out_dir')
    parser.add_argument('run_files', nargs='+')
    args = parser.parse_args(argv)
    check_unique_names(args.run_files)
    dev = dev_ids()
    out = Path(args.out_dir)
    for name in args.run_files:
        path = Path(name)
        lines = drop_dev(read_lines(path), dev, path)
        target = out / path.name
        write_lines(target, lines)
        print(f'{target}: {len(lines)} records')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run, turning input errors into a message and exit code 1."""
    try:
        return run(argv)
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
