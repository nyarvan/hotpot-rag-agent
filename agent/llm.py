"""OpenAI chat model that answers in JSON matching a schema."""
import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

DEFAULT_MODEL = 'gpt-4o-mini'
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 128
SCHEMA_HASH_CHARS = 12
REQUEST_TIMEOUT_S = 30.0
MAX_RETRIES = 2


def _schema_hash(schema: Mapping[str, Any], schema_name: str) -> str:
    """Short digest of the schema and its name, independent of key order."""
    canonical = json.dumps(schema, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(f'{schema_name}:{canonical}'.encode('utf-8'))
    return digest.hexdigest()[:SCHEMA_HASH_CHARS]


class OpenAIJsonChat:
    """`rag_eval.generate.ChatModel` that forces a strict JSON schema reply.

    The schema is part of `name`, so changing it invalidates reply caches.
    The client is built on the first call unless one is passed in.
    """

    def __init__(
            self, model: str = DEFAULT_MODEL, *,
            schema: Mapping[str, Any], schema_name: str,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            client: Any = None,
    ):
        self._model = model
        self._schema = copy.deepcopy(dict(schema))
        self._schema_name = schema_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client
        digest = _schema_hash(self._schema, schema_name)
        self.name = (
            f'openai/{model}?temperature={temperature}'
            f'&max_tokens={max_tokens}&schema={digest}'
        )

    def complete(self, system: str, user: str) -> str:
        """Return the JSON text of the model's reply."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES,
            )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': self._schema_name,
                    'schema': self._schema,
                    'strict': True,
                },
            },
        )
        if not response.choices:
            raise RuntimeError(f'{self.name} returned no choices')
        choice = response.choices[0]
        if choice.finish_reason == 'length':
            raise RuntimeError(f'{self.name} reply was truncated')
        message = choice.message
        if message.refusal:
            raise RuntimeError(f'{self.name} refused: {message.refusal}')
        if message.content is None:
            raise RuntimeError(f'{self.name} returned an empty completion')
        return message.content
