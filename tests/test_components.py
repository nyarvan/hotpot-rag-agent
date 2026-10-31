"""Tests for `RerankSearcher`, `check_caches` and the build helpers."""
from collections.abc import Sequence
from pathlib import Path

import pytest
from rag_eval.data import Question

from agent.components import (
    DENSE_MODEL,
    RERANK_CACHE_FILE,
    RerankSearcher,
    check_api_key,
    check_caches,
    seed_rerank_cache,
)


class RecordingRetriever:
    """Returns `k` titles derived from the question text; records calls."""

    def __init__(self) -> None:
        self.retrieve_calls: list[tuple[Question, int]] = []
        self.retrieve_many_calls: list[tuple[list[Question], int]] = []

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Record the call and return a ranking."""
        self.retrieve_calls.append((question, k))
        return self._ranking(question.question, k)

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Record the call and return one ranking per question."""
        self.retrieve_many_calls.append((list(questions), k))
        return [self._ranking(q.question, k) for q in questions]

    @staticmethod
    def _ranking(text: str, k: int) -> list[str]:
        return [f'{text}-{i}' for i in range(k)]


def make_question(text: str) -> Question:
    return Question(
        question_id=f'id-{text}', question=text, answer='a', type='bridge',
        gold_titles=('x', 'y'), evidence=(),
    )


def test_search_builds_question_with_query_text_and_passes_k() -> None:
    retriever = RecordingRetriever()

    ranking = RerankSearcher(retriever).search('who?', 4)

    assert len(retriever.retrieve_calls) == 1
    question, k = retriever.retrieve_calls[0]
    assert question == Question(
        question_id='', question='who?', answer='', type='',
        gold_titles=(), evidence=(),
    )
    assert k == 4
    assert ranking == ['who?-0', 'who?-1', 'who?-2', 'who?-3']


def test_prefetch_makes_exactly_one_retrieve_many_call() -> None:
    retriever = RecordingRetriever()
    questions = [make_question('q1'), make_question('q2')]

    RerankSearcher(retriever).prefetch(questions, 10)

    assert retriever.retrieve_many_calls == [(questions, 10)]
    assert retriever.retrieve_calls == []


def test_prefetched_query_is_served_without_retrieve() -> None:
    retriever = RecordingRetriever()
    searcher = RerankSearcher(retriever)
    searcher.prefetch([make_question('q1')], 10)

    ranking = searcher.search('q1', 10)

    assert ranking == [f'q1-{i}' for i in range(10)]
    assert retriever.retrieve_calls == []


def test_prefetched_query_with_smaller_k_returns_prefix() -> None:
    retriever = RecordingRetriever()
    searcher = RerankSearcher(retriever)
    searcher.prefetch([make_question('q1')], 10)

    ranking = searcher.search('q1', 3)

    assert ranking == ['q1-0', 'q1-1', 'q1-2']
    assert retriever.retrieve_calls == []


def test_larger_k_than_prefetched_goes_to_retrieve() -> None:
    retriever = RecordingRetriever()
    searcher = RerankSearcher(retriever)
    searcher.prefetch([make_question('q1')], 5)

    ranking = searcher.search('q1', 8)

    assert len(ranking) == 8
    assert [(q.question, k) for q, k in retriever.retrieve_calls] == [
        ('q1', 8),
    ]


def test_follow_up_query_goes_to_retrieve() -> None:
    retriever = RecordingRetriever()
    searcher = RerankSearcher(retriever)
    searcher.prefetch([make_question('q1')], 10)

    searcher.search('follow up', 10)

    assert [(q.question, k) for q, k in retriever.retrieve_calls] == [
        ('follow up', 10),
    ]


def make_caches(data_dir: Path) -> None:
    (data_dir / 'hotpotqa-200' / DENSE_MODEL).mkdir(parents=True)
    (data_dir / 'hotpotqa-200' / DENSE_MODEL / 'documents.npz').touch()
    (data_dir / 'rerank_cache').mkdir()
    (data_dir / 'rerank_cache' / 'BAAI__bge-reranker-v2-m3.json').touch()
    (data_dir / 'answer_cache').mkdir()
    (data_dir / 'hf_cache').mkdir()


REQUIRED = [
    Path('hotpotqa-200') / DENSE_MODEL / 'documents.npz',
    Path('rerank_cache') / 'BAAI__bge-reranker-v2-m3.json',
    Path('answer_cache'),
    Path('hf_cache'),
]


def test_check_caches_passes_on_complete_tree(tmp_path: Path) -> None:
    make_caches(tmp_path)

    check_caches(tmp_path)


def test_check_caches_names_missing_data_dir(tmp_path: Path) -> None:
    missing = tmp_path / 'nope'

    with pytest.raises(FileNotFoundError, match='nope') as error:
        check_caches(missing)

    assert 'recompute' in str(error.value)


@pytest.mark.parametrize('relative', REQUIRED, ids=str)
def test_check_caches_names_each_missing_path(
        tmp_path: Path, relative: Path,
) -> None:
    make_caches(tmp_path)
    target = tmp_path / relative
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(FileNotFoundError) as error:
        check_caches(tmp_path)

    assert str(target) in str(error.value)
    assert 'recompute' in str(error.value)


def make_rerank_cache(base: Path, content: str) -> Path:
    path = base / 'rerank_cache' / RERANK_CACHE_FILE
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding='utf-8')
    return path


def test_seed_rerank_cache_copies_when_agent_copy_is_missing(
        tmp_path: Path,
) -> None:
    harness, agent = tmp_path / 'harness', tmp_path / 'agent'
    make_rerank_cache(harness, '{"a": 1}')

    cache_dir = seed_rerank_cache(harness, agent)

    assert cache_dir == agent / 'rerank_cache'
    copied = cache_dir / RERANK_CACHE_FILE
    assert copied.read_text(encoding='utf-8') == '{"a": 1}'


def test_seed_rerank_cache_keeps_an_existing_agent_copy(
        tmp_path: Path,
) -> None:
    harness, agent = tmp_path / 'harness', tmp_path / 'agent'
    make_rerank_cache(harness, 'harness')
    existing = make_rerank_cache(agent, 'agent')

    seed_rerank_cache(harness, agent)

    assert existing.read_text(encoding='utf-8') == 'agent'


def test_seed_rerank_cache_leaves_the_harness_file_untouched(
        tmp_path: Path,
) -> None:
    harness, agent = tmp_path / 'harness', tmp_path / 'agent'
    source = make_rerank_cache(harness, 'harness')

    cache_dir = seed_rerank_cache(harness, agent)
    (cache_dir / RERANK_CACHE_FILE).write_text('changed', encoding='utf-8')

    assert source.read_text(encoding='utf-8') == 'harness'


def test_check_api_key_passes_when_set() -> None:
    check_api_key({'OPENAI_API_KEY': 'sk-secret'})


@pytest.mark.parametrize('environ', [{}, {'OPENAI_API_KEY': ''}], ids=str)
def test_check_api_key_raises_when_missing_or_empty(
        environ: dict[str, str],
) -> None:
    with pytest.raises(RuntimeError, match='OPENAI_API_KEY is not set'):
        check_api_key(environ)
