"""Count records of two run files that match on every RunRecord field.

Run from the repo root:
    venv/bin/python -m scripts.same_records AGENT.jsonl BASELINE.jsonl

Exits 1 when a shared record differs or the agent has ids the baseline lacks.
"""
import argparse
import sys
from collections.abc import Sequence

from rag_eval.records import RunRecord, read_records

MAX_SHOWN = 10


def main(argv: Sequence[str] | None = None) -> int:
    """Print matches and the first differing fields; 1 on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('agent')
    parser.add_argument('baseline')
    args = parser.parse_args(argv)
    fields = list(RunRecord.model_fields)
    agent = {r.question_id: r for r in read_records(args.agent)}
    base = {r.question_id: r for r in read_records(args.baseline)}
    shared = [qid for qid in agent if qid in base]
    diffs = []
    for qid in shared:
        a = agent[qid].model_dump(include=set(fields))
        b = base[qid].model_dump(include=set(fields))
        bad = [f for f in fields if a[f] != b[f]]
        if bad:
            diffs.append((qid, bad))
    print(f'agent {len(agent)}, baseline {len(base)}, shared {len(shared)}')
    print(f'identical: {len(shared) - len(diffs)}/{len(shared)}')
    for qid, bad in diffs[:MAX_SHOWN]:
        print(f'  {qid}: {", ".join(bad)}')
    return 0 if not diffs and len(shared) == len(agent) else 1


if __name__ == '__main__':
    sys.exit(main())
