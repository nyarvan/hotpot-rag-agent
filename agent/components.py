"""Real dependencies of the agent: retriever, decider and answerer.

Embeddings, answers and the dataset are cached by the shared harness under
`data_dir`; `build_components` refuses to start if a cache is missing,
because the alternative is silently recomputing embeddings, rerank scores
or answers. The agent keeps its own rerank and decide caches under
`agent_data_dir`, so follow-up queries never rewrite harness files.
"""
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_eval.data import (
    DEFAULT_N,
    Document,
    HotpotSubset,
    Question,
    load_hotpotqa,
)
from rag_eval.generate import Generator, OpenAIChat
from rag_eval.rerank import RerankedRetriever
from rag_eval.retrieve import DenseRetriever

from .decide import DECIDE_SCHEMA, SCHEMA_NAME, LLMDecider
from .llm import OpenAIJsonChat
from .nodes import Answerer, Decider

DEFAULT_DATA_DIR = Path('../rag-eval-harness/data')
AGENT_DATA_DIR = Path('data')
DENSE_MODEL = 'bge_m3'
CANDIDATES_K = 30
DEFAULT_CHAT_MODEL = 'gpt-4o-mini'
RERANK_CACHE_FILE = 'BAAI__bge-reranker-v2-m3.json'


class QuestionRetriever(Protocol):
    """Retriever over `Question` objects (matches the harness retrievers)."""

    def retrieve(self, question: Question, k: int) -> list[str]:
        """Return the top-k ids for one question."""
        ...

    def retrieve_many(
            self, questions: Sequence[Question], k: int,
    ) -> list[list[str]]:
        """Return the top-k ids for each question, encoded as one batch."""
        ...


class RerankSearcher:
    """Searches by query text, serving prefetched first-hop rankings.

    The baseline encodes all questions in one batch; encoding them one by
    one can reorder near-ties, so `max_hops=1` would no longer reproduce it.
    A new `prefetch` replaces any earlier one.
    """

    def __init__(self, retriever: QuestionRetriever) -> None:
        self._retriever = retriever
        self._prefetched: dict[str, tuple[int, list[str]]] = {}

    def prefetch(self, questions: Sequence[Question], k: int) -> None:
        """Rank all questions with a single batched retriever call."""
        rankings = self._retriever.retrieve_many(questions, k)
        self._prefetched = {
            question.question: (k, list(ranking))
            for question, ranking in zip(questions, rankings, strict=True)
        }

    def search(self, query: str, k: int) -> list[str]:
        """Return the top-k ids, from the prefetch when it is deep enough."""
        cached = self._prefetched.get(query)
        if cached is not None and cached[0] >= k:
            return cached[1][:k]
        question = Question(
            question_id='', question=query, answer='', type='',
            gold_titles=(), evidence=(),
        )
        return self._retriever.retrieve(question, k)


@dataclass(frozen=True)
class Components:
    """Everything `build_graph` and `run_agent` need."""

    subset: HotpotSubset
    documents: Mapping[str, Document]
    searcher: RerankSearcher
    decider: Decider
    answerer: Answerer


def check_caches(data_dir: str | Path) -> None:
    """Raise `FileNotFoundError` if a shared harness cache is missing."""
    data_dir = Path(data_dir)
    required = (
        data_dir,
        data_dir / 'hotpotqa-200' / DENSE_MODEL / 'documents.npz',
        data_dir / 'rerank_cache' / RERANK_CACHE_FILE,
        data_dir / 'answer_cache',
        data_dir / 'hf_cache',
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f'Missing cache: {path}. Running without it would '
                'recompute it (embeddings, rerank scores or answers).'
            )


def check_api_key(environ: Mapping[str, str]) -> None:
    """Raise `RuntimeError` if `OPENAI_API_KEY` is unset or empty."""
    if not environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is not set; load .env first')


def seed_rerank_cache(
        harness_dir: str | Path, agent_dir: str | Path,
) -> Path:
    """Copy the harness rerank cache into the agent dir once.

    Returns the agent's rerank cache directory. An existing agent copy is
    never overwritten, and the harness file is never written to.
    """
    source = Path(harness_dir) / 'rerank_cache' / RERANK_CACHE_FILE
    cache_dir = Path(agent_dir) / 'rerank_cache'
    target = cache_dir / RERANK_CACHE_FILE
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return cache_dir


def build_components(
        data_dir: str | Path = DEFAULT_DATA_DIR, *,
        agent_data_dir: str | Path = AGENT_DATA_DIR,
        chat_model: str = DEFAULT_CHAT_MODEL,
) -> Components:
    """Load the subset and wire the cached retriever, decider and answerer."""
    check_caches(data_dir)
    check_api_key(os.environ)
    data_dir = Path(data_dir)
    agent_data_dir = Path(agent_data_dir)
    subset = load_hotpotqa(cache_dir=data_dir / 'hf_cache')
    dense = DenseRetriever(
        DENSE_MODEL, subset.documents, subset_size=DEFAULT_N,
        data_dir=data_dir,
    )
    reranked = RerankedRetriever(
        dense, subset.documents, candidates_k=CANDIDATES_K,
        cache_dir=seed_rerank_cache(data_dir, agent_data_dir),
    )
    json_chat = OpenAIJsonChat(
        chat_model, schema=DECIDE_SCHEMA, schema_name=SCHEMA_NAME,
    )
    return Components(
        subset=subset,
        documents={d.doc_id: d for d in subset.documents},
        searcher=RerankSearcher(reranked),
        decider=LLMDecider(json_chat, agent_data_dir / 'decide_cache'),
        answerer=Generator(
            OpenAIChat(chat_model), data_dir / 'answer_cache',
        ),
    )
