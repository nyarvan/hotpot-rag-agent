import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.llm import OpenAIJsonChat

SCHEMA = {
    'type': 'object',
    'properties': {'ok': {'type': 'boolean'}},
    'required': ['ok'],
    'additionalProperties': False,
}


def make_client(
        content='{"ok": true}', refusal=None, calls=None,
        finish_reason='stop', choices=True,
) -> None:
    """Fake OpenAI client returning one message and recording requests."""

    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        message = SimpleNamespace(content=content, refusal=refusal)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice] if choices else [])

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def make_chat(client=None, **overrides):
    options = {'schema': SCHEMA, 'schema_name': 'reply', 'client': client}
    options.update(overrides)
    return OpenAIJsonChat(**options)


def test_complete_returns_message_content() -> None:
    chat = make_chat(make_client('{"ok": false}'))

    assert chat.complete('sys', 'usr') == '{"ok": false}'


def test_request_carries_strict_json_schema_format() -> None:
    calls = []
    chat = make_chat(make_client(calls=calls))

    chat.complete('sys', 'usr')

    assert calls[0]['response_format'] == {
        'type': 'json_schema',
        'json_schema': {'name': 'reply', 'schema': SCHEMA, 'strict': True},
    }


def test_request_has_zero_temperature_model_and_both_messages() -> None:
    calls = []
    chat = make_chat(make_client(calls=calls))

    chat.complete('sys', 'usr')

    assert calls[0]['model'] == 'gpt-4o-mini'
    assert calls[0]['temperature'] == 0.0
    assert calls[0]['max_tokens'] == 128
    assert calls[0]['messages'] == [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'usr'},
    ]


def test_name_changes_with_model() -> None:
    assert make_chat().name != make_chat(model='gpt-4o').name


def test_name_changes_with_temperature() -> None:
    assert make_chat().name != make_chat(temperature=0.5).name


def test_name_changes_with_schema() -> None:
    other = {**SCHEMA, 'required': []}

    assert make_chat().name != make_chat(schema=other).name


def test_name_changes_with_schema_name() -> None:
    assert make_chat().name != make_chat(schema_name='other').name


def test_name_is_stable_for_same_settings() -> None:
    assert make_chat().name == make_chat().name


def test_name_ignores_schema_key_order() -> None:
    reordered = dict(reversed(list(SCHEMA.items())))

    assert make_chat().name == make_chat(schema=reordered).name


def test_refusal_raises_naming_the_model() -> None:
    chat = make_chat(make_client(content=None, refusal='no'))

    with pytest.raises(RuntimeError, match='gpt-4o-mini'):
        chat.complete('sys', 'usr')


def test_empty_content_raises() -> None:
    chat = make_chat(make_client(content=None))

    with pytest.raises(RuntimeError, match='empty'):
        chat.complete('sys', 'usr')


def test_client_is_not_built_at_construction(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError('client built too early')

    monkeypatch.setattr('openai.OpenAI', boom)

    make_chat()


def test_client_is_built_lazily_on_first_call(monkeypatch) -> None:
    built = []

    def factory(**kwargs):
        built.append(True)
        return make_client()

    monkeypatch.setattr('openai.OpenAI', factory)
    chat = make_chat()

    chat.complete('sys', 'usr')
    chat.complete('sys', 'usr')

    assert built == [True]


def test_client_is_built_with_timeout_and_retry_limit(monkeypatch) -> None:
    kwargs_seen = []

    def factory(**kwargs):
        kwargs_seen.append(kwargs)
        return make_client()

    monkeypatch.setattr('openai.OpenAI', factory)

    make_chat().complete('sys', 'usr')

    assert kwargs_seen == [{'timeout': 30.0, 'max_retries': 2}]


def test_truncated_reply_raises() -> None:
    chat = make_chat(make_client('{"ok": tr', finish_reason='length'))

    with pytest.raises(RuntimeError, match='truncated') as error:
        chat.complete('sys', 'usr')

    assert chat.name in str(error.value)


def test_no_choices_raises() -> None:
    chat = make_chat(make_client(choices=False))

    with pytest.raises(RuntimeError, match='no choices') as error:
        chat.complete('sys', 'usr')

    assert chat.name in str(error.value)


def test_name_is_pinned_to_protect_the_paid_cache() -> None:
    schema = {'type': 'object', 'properties': {}}
    canonical = json.dumps(schema, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(f'decide:{canonical}'.encode('utf-8'))

    chat = OpenAIJsonChat(
        'gpt-4o-mini', schema=schema, schema_name='decide',
    )

    assert chat.name == (
        'openai/gpt-4o-mini?temperature=0.0&max_tokens=128'
        f'&schema={digest.hexdigest()[:12]}'
    )


def test_later_change_to_the_passed_schema_does_not_affect_requests() -> None:
    calls = []
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}}
    chat = make_chat(make_client(calls=calls), schema=schema)
    schema['properties']['ok']['type'] = 'string'

    chat.complete('sys', 'usr')

    sent = calls[0]['response_format']['json_schema']['schema']
    assert sent['properties']['ok']['type'] == 'boolean'
