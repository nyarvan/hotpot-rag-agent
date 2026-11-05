"""Split run files by question type into OUT/{bridge,comparison}/.

Lines are copied verbatim; the type comes from the HotpotQA subset. With
--heldout the 25 dev questions are dropped too.

Run from the repo root (offline):
    HF_HUB_OFFLINE=1 venv/bin/python -m scripts.split_by_type \
        --out runs/by_type [--heldout] RUN_FILE [RUN_FILE ...]
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .common import (
    QUESTION_TYPES,
    check_unique_names,
    dev_ids,
    drop_dev,
    line_id,
    question_types,
    read_lines,
    type_of,
    write_lines,
)


def run(argv: Sequence[str] | None) -> int:
    """Write OUT/<type>/<run file name> for every type and file."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--heldout', action='store_true')
    parser.add_argument('run_files', nargs='+')
    args = parser.parse_args(argv)
    check_unique_names(args.run_files)
    types = question_types()
    dev = dev_ids() if args.heldout else set()
    for name in args.run_files:
        path = Path(name)
        by_type: dict[str, list[str]] = {t: [] for t in QUESTION_TYPES}
        lines = read_lines(path)
        if args.heldout:
            lines = drop_dev(lines, dev, path)
        for line in lines:
            question_type = type_of(types, line_id(line, path))
            by_type[question_type].append(line)
        for question_type, typed in by_type.items():
            target = Path(args.out) / question_type / path.name
            write_lines(target, typed)
            print(f'{target}: {len(typed)} records')
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
