import json
import logging

import httpx
import openai
import pytest
from rag_eval.data import Document
from rag_eval.generate import cache_key

from agent.decide import (
    MAX_FOLLOW_UP_CHARS,
    SYSTEM_PROMPT,
    LLMDecider,
    build_decide_prompt,
)
from agent.nodes import Decision

CONTEXTS = (
    Document('Alpha', 'Alpha', 'alpha text'),
    Document('Beta', 'Beta', 'beta text'),
)
QUESTION = 'Who directed the film?'


def reply(enough=False, follow_up='Director of Alpha'):
    return json.dumps({'enough': enough, 'follow_up': follow_up})


class FakeChat:
    """ChatModel with a fixed reply (or error) that records every call."""

    def __init__(self, text=None, name='fake', error=None):
        self.name = name
        self.text = reply() if text is None else text
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.text


def decide(chat, tmp_path):
    return LLMDecider(chat, tmp_path).decide(QUESTION, CONTEXTS)


def test_valid_reply_becomes_decision(tmp_path) -> None:
    result = decide(FakeChat(reply(False, 'Director of Alpha')), tmp_path)

    assert result == Decision(enough=False, follow_up='Director of Alpha')


def test_enough_true_clears_follow_up(tmp_path) -> None:
    result = decide(FakeChat(reply(True, 'ignored')), tmp_path)

    assert result == Decision(enough=True, follow_up='')


def test_follow_up_is_stripped(tmp_path) -> None:
    result = decide(FakeChat(reply(False, '  Director of Alpha \n')), tmp_path)

    assert result.follow_up == 'Director of Alpha'


def test_follow_up_whitespace_is_collapsed(tmp_path) -> None:
    chat = FakeChat(reply(False, 'Director\n of\t\tAlpha'))

    result = decide(chat, tmp_path)

    assert result.follow_up == 'Director of Alpha'


def test_follow_up_drops_non_printable_characters(tmp_path) -> None:
    chat = FakeChat(reply(False, 'Direc\u200btor of Alpha'))

    result = decide(chat, tmp_path)

    assert result.follow_up == 'Director of Alpha'


def test_follow_up_is_capped(tmp_path) -> None:
    long_text = 'x' * (MAX_FOLLOW_UP_CHARS + 100)

    result = decide(FakeChat(reply(False, long_text)), tmp_path)

    assert result.follow_up == 'x' * MAX_FOLLOW_UP_CHARS


def test_invalid_json_falls_back_and_warns(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger='agent.decide'):
        result = decide(FakeChat('not json'), tmp_path)

    assert result == Decision(enough=True)
    assert 'Invalid decide reply' in caplog.text
    assert 'not json' in caplog.text


@pytest.mark.parametrize('text', [
    '{"enough": false}',
    '{"follow_up": "q"}',
    '{"enough": false, "follow_up": "q", "extra": 1}',
    '{"enough": "yes", "follow_up": "q"}',
    '{"enough": 1, "follow_up": "q"}',
    '{"enough": false, "follow_up": 5}',
    '[]',
])
def test_malformed_reply_falls_back_without_caching(tmp_path, text) -> None:
    result = decide(FakeChat(text), tmp_path)

    assert result == Decision(enough=True)
    assert list(tmp_path.iterdir()) == []


def test_runtime_error_falls_back_warns_and_writes_no_cache(
        tmp_path, caplog,
) -> None:
    chat = FakeChat(error=RuntimeError('boom'))
    decider = LLMDecider(chat, tmp_path)

    with caplog.at_level(logging.WARNING, logger='agent.decide'):
        result = decider.decide(QUESTION, CONTEXTS)

    assert result == Decision(enough=True)
    assert 'Decide model refused or returned nothing' in caplog.text
    assert 'boom' in caplog.text
    assert list(tmp_path.iterdir()) == []
    assert decider.fallbacks == 1


def test_runtime_error_text_is_truncated_in_log(tmp_path, caplog) -> None:
    chat = FakeChat(error=RuntimeError('a' * 200 + 'TAIL'))

    with caplog.at_level(logging.WARNING, logger='agent.decide'):
        decide(chat, tmp_path)

    assert 'a' * 200 in caplog.text
    assert 'TAIL' not in caplog.text


def test_invalid_reply_log_keeps_only_the_tail(tmp_path, caplog) -> None:
    chat = FakeChat('HEAD' + 'x' * 80)

    with caplog.at_level(logging.WARNING, logger='agent.decide'):
        decide(chat, tmp_path)

    assert 'HEAD' not in caplog.text
    assert 'x' * 80 in caplog.text


@pytest.mark.parametrize('error', [
    openai.APIConnectionError(
        request=httpx.Request(
            'POST', 'https://api.openai.com/v1/chat/completions',
        ),
    ),
    ValueError('custom'),
], ids=['openai-connection', 'plain-exception'])
def test_non_runtime_errors_propagate_without_logging(
        tmp_path, caplog, error,
) -> None:
    decider = LLMDecider(FakeChat(error=error), tmp_path)

    with caplog.at_level(logging.WARNING, logger='agent.decide'):
        with pytest.raises(type(error)):
            decider.decide(QUESTION, CONTEXTS)

    assert caplog.text == ''
    assert decider.fallbacks == 0
    assert list(tmp_path.iterdir()) == []


def test_fallbacks_start_at_zero_and_count_each_fallback(tmp_path) -> None:
    decider = LLMDecider(FakeChat('not json'), tmp_path)
    assert decider.fallbacks == 0

    decider.decide(QUESTION, CONTEXTS)
    decider.decide('Another?', CONTEXTS)

    assert decider.fallbacks == 2


def test_valid_reply_does_not_count_as_fallback(tmp_path) -> None:
    decider = LLMDecider(FakeChat(), tmp_path)

    decider.decide(QUESTION, CONTEXTS)

    assert decider.fallbacks == 0


@pytest.mark.parametrize('content', [
    'not json at all', '{}', '[]', '{"reply": 5}',
], ids=['garbage', 'empty-object', 'list', 'non-str-reply'])
def test_corrupt_cache_file_is_replaced_after_asking_again(
        tmp_path, caplog, content,
) -> None:
    chat = FakeChat(reply(False, 'q'))
    LLMDecider(chat, tmp_path).decide(QUESTION, CONTEXTS)
    path = next(tmp_path.iterdir())
    path.write_text(content, encoding='utf-8')
    chat.calls.clear()

    with caplog.at_level(logging.WARNING, logger='agent.decide'):
        result = decide(chat, tmp_path)

    assert result == Decision(enough=False, follow_up='q')
    assert len(chat.calls) == 1
    assert 'Unreadable decide cache' in caplog.text
    assert json.loads(path.read_text(encoding='utf-8'))['reply'] == chat.text


def test_unreadable_cache_path_is_asked_again(tmp_path) -> None:
    chat = FakeChat()
    LLMDecider(chat, tmp_path).decide(QUESTION, CONTEXTS)
    path = next(tmp_path.iterdir())
    path.write_bytes(b'\xff\xfe\x00')
    chat.calls.clear()

    decide(chat, tmp_path)

    assert len(chat.calls) == 1


def test_same_inputs_are_served_from_cache(tmp_path) -> None:
    chat = FakeChat()

    first = decide(chat, tmp_path)
    second = decide(chat, tmp_path)

    assert first == second
    assert len(chat.calls) == 1


def test_other_chat_name_is_not_served_from_cache(tmp_path) -> None:
    decide(FakeChat(name='one'), tmp_path)
    other = FakeChat(reply(True, ''), name='two')

    result = decide(other, tmp_path)

    assert len(other.calls) == 1
    assert result.enough is True


def test_cache_stores_raw_reply_and_inputs(tmp_path) -> None:
    chat = FakeChat(reply(False, '  padded  '))
    decide(chat, tmp_path)
    user = build_decide_prompt(QUESTION, CONTEXTS)
    path = tmp_path / f'{cache_key("fake", SYSTEM_PROMPT, user)}.json'

    record = json.loads(path.read_text(encoding='utf-8'))

    assert record == {
        'model': 'fake',
        'system': SYSTEM_PROMPT,
        'user': user,
        'reply': chat.text,
    }


def test_cached_reply_is_parsed_again(tmp_path) -> None:
    decide(FakeChat(reply(False, 'q')), tmp_path)
    path = next(tmp_path.iterdir())
    record = json.loads(path.read_text(encoding='utf-8'))
    record['reply'] = 'garbage'
    path.write_text(json.dumps(record), encoding='utf-8')
    chat = FakeChat()

    result = decide(chat, tmp_path)

    assert result == Decision(enough=True)
    assert chat.calls == []


def test_cache_dir_is_created_without_temp_files(tmp_path) -> None:
    cache_dir = tmp_path / 'nested' / 'cache'

    LLMDecider(FakeChat(), cache_dir).decide(QUESTION, CONTEXTS)

    names = [path.name for path in cache_dir.iterdir()]
    assert len(names) == 1 and names[0].endswith('.json')


def test_chat_receives_system_and_user_prompts(tmp_path) -> None:
    chat = FakeChat()

    decide(chat, tmp_path)

    assert chat.calls == [
        (SYSTEM_PROMPT, build_decide_prompt(QUESTION, CONTEXTS)),
    ]


def test_user_prompt_lists_passages_in_rank_order_then_question() -> None:
    prompt = build_decide_prompt(QUESTION, CONTEXTS)

    first = prompt.index('Title: Alpha\nText: alpha text')
    second = prompt.index('Title: Beta\nText: beta text')
    assert prompt.startswith('Passages:')
    assert first < second < prompt.index(f'Question: {QUESTION}')


def test_system_prompt_forbids_outside_knowledge() -> None:
    text = SYSTEM_PROMPT.lower()

    assert 'only from the passages' in text
    assert 'outside knowledge' in text


def test_system_prompt_covers_comparison_questions() -> None:
    text = SYSTEM_PROMPT.lower()

    assert 'comparison' in text
    assert 'a passage about each' in text
    assert 'no passage at all' in text


def test_system_prompt_asks_for_bridge_entity_name() -> None:
    text = SYSTEM_PROMPT.lower()

    assert 'bridge' in text
    assert 'name of that entity' in text
    assert 'incidental' in text
    assert 'never restate the question' in text


def test_system_prompt_stays_compact() -> None:
    assert len(SYSTEM_PROMPT) < 1200
