"""LLM judge: is the retrieved context enough, or what to search next."""
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError
from rag_eval.data import Document
from rag_eval.generate import ChatModel, cache_key, format_context

from .nodes import Decision

logger = logging.getLogger(__name__)

SCHEMA_NAME = 'decide'
DECIDE_SCHEMA = {
    'type': 'object',
    'properties': {
        'enough': {'type': 'boolean'},
        'follow_up': {'type': 'string'},
    },
    'required': ['enough', 'follow_up'],
    'additionalProperties': False,
}
MAX_FOLLOW_UP_CHARS = 500
DEFAULT_CACHE_DIR = 'data/decide_cache'
LOG_TAIL_CHARS = 80
ERROR_LOG_CHARS = 200

SYSTEM_PROMPT = (
    'You check whether the passages contain every fact needed to answer a '
    'multi-hop question. Answer in JSON. '
    'If they do, set enough=true and follow_up="". '
    'If a fact is missing and the passages name the entity it is about '
    '(the bridge, for example the person, film or company that a found '
    'passage mentions), set enough=false and set follow_up to a short '
    'search query naming that entity and the missing attribute, not a '
    'restatement of the question. '
    'If nothing better can be searched, set enough=true.'
)


class DecideReply(BaseModel):
    """The JSON reply the model must produce."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    enough: StrictBool
    follow_up: str


def build_decide_prompt(question: str, contexts: Sequence[Document]) -> str:
    """Passages in rank order, then the question."""
    passages = '\n\n'.join(format_context(d) for d in contexts)
    return f'Passages:\n{passages}\n\nQuestion: {question}'


def _clean_follow_up(text: str) -> str:
    """Collapse whitespace, drop non-printable characters, cap the length."""
    printable = ''.join(c for c in text if c.isprintable() or c.isspace())
    return ' '.join(printable.split())[:MAX_FOLLOW_UP_CHARS]


def _to_decision(reply: DecideReply) -> Decision:
    """Decision with a normalised follow-up (empty when enough)."""
    if reply.enough:
        return Decision(enough=True)
    return Decision(enough=False, follow_up=_clean_follow_up(reply.follow_up))


def _read(path: Path) -> str | None:
    """Cached raw reply, or None when absent or unreadable."""
    try:
        text = json.loads(path.read_text(encoding='utf-8'))['reply']
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        text = None
    if not isinstance(text, str):
        logger.warning(
            'Unreadable decide cache %s, asking the model again', path.name,
        )
        return None
    return text


class LLMDecider:
    """`agent.nodes.Decider` backed by a chat model, with a disk cache.

    The raw reply is cached and parsed again on every hit, so parser fixes
    also apply to replies that were paid for earlier. A refusal, an empty or
    truncated reply, or an invalid one means "answer now" and is counted in
    `fallbacks`; API errors propagate.
    """

    def __init__(
            self, chat: ChatModel, cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._chat = chat
        self._cache_dir = Path(cache_dir)
        self.fallbacks = 0

    def decide(self, question: str, contexts: Sequence[Document]) -> Decision:
        """Return the model's decision, or "enough" if it cannot be read."""
        user = build_decide_prompt(question, contexts)
        key = cache_key(self._chat.name, SYSTEM_PROMPT, user)
        path = self._cache_dir / f'{key}.json'
        text = _read(path)
        cached = text is not None
        if text is None:
            try:
                text = self._chat.complete(SYSTEM_PROMPT, user)
            except RuntimeError as error:
                logger.warning(
                    'Decide model refused or returned nothing, '
                    'answering now: %s', str(error)[:ERROR_LOG_CHARS],
                )
                return self._fallback()
        try:
            reply = DecideReply.model_validate_json(text)
        except ValidationError:
            logger.warning(
                'Invalid decide reply, answering now: %r',
                text[-LOG_TAIL_CHARS:],
            )
            return self._fallback()
        if not cached:
            self._write(path, user, text)
        return _to_decision(reply)

    def _fallback(self) -> Decision:
        """Count the fallback and tell the graph to answer now."""
        self.fallbacks += 1
        return Decision(enough=True)

    def _write(self, path: Path, user: str, reply: str) -> None:
        """Store the reply with its inputs; the file appears all at once."""
        record = {
            'model': self._chat.name,
            'system': SYSTEM_PROMPT,
            'user': user,
            'reply': reply,
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        os.replace(temporary, path)
