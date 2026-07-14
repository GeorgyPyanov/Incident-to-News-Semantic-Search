from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from retrieval.embeddings import OpenAIEmbeddingClient


class OpenAIEmbeddingClientTests(unittest.TestCase):
    def test_embed_texts_trims_inputs_and_preserves_embedding_order(self) -> None:
        fake_response = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )

        with patch("retrieval.embeddings.OpenAI") as openai_cls:
            openai_instance = openai_cls.return_value
            openai_instance.embeddings.create.return_value = fake_response

            client = OpenAIEmbeddingClient(api_key="test-key", model="test-model")
            vectors = client.embed_texts(["  first prompt  ", "\nsecond prompt\t"])

        openai_cls.assert_called_once_with(api_key="test-key")
        openai_instance.embeddings.create.assert_called_once_with(
            model="test-model",
            input=["first prompt", "second prompt"],
        )
        self.assertEqual([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], vectors)

    def test_embed_text_returns_first_embedding(self) -> None:
        fake_response = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])])

        with patch("retrieval.embeddings.OpenAI") as openai_cls:
            openai_instance = openai_cls.return_value
            openai_instance.embeddings.create.return_value = fake_response

            client = OpenAIEmbeddingClient(api_key=None)
            vector = client.embed_text(" incident log ")

        self.assertEqual([1.0, 2.0, 3.0], vector)

    def test_raises_when_openai_returns_mismatched_embedding_count(self) -> None:
        fake_response = SimpleNamespace(data=[])

        with patch("retrieval.embeddings.OpenAI") as openai_cls:
            openai_instance = openai_cls.return_value
            openai_instance.embeddings.create.return_value = fake_response

            client = OpenAIEmbeddingClient()

            with self.assertRaises(RuntimeError):
                client.embed_texts(["one prompt"])


if __name__ == "__main__":
    unittest.main()
