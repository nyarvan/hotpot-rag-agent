"""Paired comparison of two RAGAS score files by question_id.

The first file is the baseline: mean diff = second minus first, as in
`rag-eval compare`. Questions where either value is null are skipped (and
counted). Per metric: both means, mean diff with a 95% paired-bootstrap CI,
better/worse counts and the exact two-sided sign test.

Run from the repo root (offline):
    HF_HUB_OFFLINE=1 venv/bin/python -m scripts.compare_scores \
        ../rag-eval-harness/runs/ragas_dense_rerank_200.jsonl \
        ../rag-eval-harness/runs/ragas_dense_rerank_cot_200.jsonl \
        [--heldout] [--type bridge|comparison]
"""
import argparse
import sys
from collections.abc import Mapping, Sequence

import numpy as np
from rag_eval.compare import EPS, paired_bootstrap, sign_test_p

from .common import (
    QUESTION_TYPES,
    dev_ids,
    question_types,
    read_rows,
    type_of,
)

ID_FIELD = 'question_id'


def common_metrics(
        first: Mapping[str, Mapping], second: Mapping[str, Mapping],
        ids: Sequence[str],
) -> list[str]:
    """Metrics present in every shared row of both files.

    Metrics seen in some rows only, or in one file only, are reported as
    dropped (per file) instead of being compared on a partial set.
    """
    def keys(rows: Mapping[str, Mapping]) -> list[set[str]]:
        return [set(rows[q]) - {ID_FIELD} for q in ids]

    sets_first, sets_second = keys(first), keys(second)
    common = set.intersection(*sets_first, *sets_second)
    for label, sets in (('first', sets_first), ('second', sets_second)):
        dropped = sorted(set.union(*sets) - common)
        if dropped:
            print(f'dropped from {label} file: {", ".join(dropped)}')
    return sorted(common, key=lambda m: list(first[ids[0]]).index(m))


def main(argv: Sequence[str] | None = None) -> int:
    """Print one row per metric present in both files."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('baseline')
    parser.add_argument('other')
    parser.add_argument('--heldout', action='store_true')
    parser.add_argument('--type', choices=QUESTION_TYPES)
    args = parser.parse_args(argv)
    first, second = read_rows(args.baseline), read_rows(args.other)
    ids = [qid for qid in first if qid in second]
    if args.heldout:
        skip = dev_ids()
        ids = [qid for qid in ids if qid not in skip]
    if args.type:
        types = question_types()
        ids = [qid for qid in ids if type_of(types, qid) == args.type]
    if not ids:
        print('no shared questions', file=sys.stderr)
        return 1
    metrics = common_metrics(first, second, ids)
    print(f'{len(ids)} shared questions; diff = second - first')
    header = (f'{"metric":<22}{"n":>5}{"skipped":>8}{"first":>8}{"second":>8}'
              f'{"diff":>9}  {"95% CI":<20}{"better/worse":>13}{"p":>8}')
    print(header)
    for metric in metrics:
        pairs = [
            (first[q][metric], second[q][metric]) for q in ids
            if first[q].get(metric) is not None
            and second[q].get(metric) is not None
        ]
        skipped = len(ids) - len(pairs)
        if not pairs:
            print(f'{metric}: skipped, no question has a value in both files')
            continue
        a = np.array([p[0] for p in pairs], dtype=float)
        b = np.array([p[1] for p in pairs], dtype=float)
        mean, low, high = paired_bootstrap(b, a)
        better = int((b - a > EPS).sum())
        worse = int((a - b > EPS).sum())
        interval = f'[{low:+.4f}, {high:+.4f}]'
        print(f'{metric:<22}{len(pairs):>5}{skipped:>8}{a.mean():>8.3f}'
              f'{b.mean():>8.3f}{mean:>+9.4f}  {interval:<20}'
              f'{f"{better}/{worse}":>13}{sign_test_p(better, worse):>8.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
