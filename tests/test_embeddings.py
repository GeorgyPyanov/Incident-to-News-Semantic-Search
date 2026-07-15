from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from retrieval.embeddings import (
    HashingEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    build_embedding_client,
    build_document_text,
    build_query_text,
    validate_embedding_dimension,
)


class SentenceTransformerEmbeddingClientTests(unittest.TestCase):
    def test_embed_texts_trims_inputs_and_preserves_embedding_order(self) -> None:
        fake_model = SimpleNamespace(
            encode=lambda texts, **kwargs: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        with patch("retrieval.embeddings.SentenceTransformer", return_value=fake_model) as transformer_cls:
            client = SentenceTransformerEmbeddingClient(model="test-model")
            vectors = client.embed_texts(["  first prompt  ", "\nsecond prompt\t"])

        transformer_cls.assert_called_once_with("test-model", device=None)
        self.assertEqual([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], vectors)

    def test_embed_text_returns_first_embedding(self) -> None:
        fake_model = SimpleNamespace(
            encode=lambda texts, **kwargs: [[1.0, 2.0, 3.0]],
        )

        with patch("retrieval.embeddings.SentenceTransformer", return_value=fake_model):
            client = SentenceTransformerEmbeddingClient()
            vector = client.embed_text(" incident log ")

        self.assertEqual([1.0, 2.0, 3.0], vector)

    def test_dynamic_quantization_is_applied_when_requested(self) -> None:
        fake_model = SimpleNamespace(
            encode=lambda texts, **kwargs: [[0.1, 0.2, 0.3]],
        )

        with (
            patch("retrieval.embeddings.SentenceTransformer", return_value=fake_model),
            patch("retrieval.embeddings._apply_dynamic_quantization") as quantize,
        ):
            client = SentenceTransformerEmbeddingClient(model="test-model", quantization="dynamic")

        quantize.assert_called_once_with(fake_model)
        self.assertEqual("test-model@dynamic", client.model_name)

    def test_build_client_passes_quantization_to_sentence_transformer(self) -> None:
        fake_model = SimpleNamespace(
            encode=lambda texts, **kwargs: [[0.1, 0.2, 0.3]],
        )

        with (
            patch("retrieval.embeddings.SentenceTransformer", return_value=fake_model),
            patch("retrieval.embeddings._apply_dynamic_quantization") as quantize,
        ):
            client = build_embedding_client(
                backend="sentence-transformer",
                model="test-model",
                quantization="dynamic",
            )

        quantize.assert_called_once_with(fake_model)
        self.assertEqual("test-model@dynamic", client.model_name)

    def test_hashing_fallback_respects_configured_dimension(self) -> None:
        client = HashingEmbeddingClient(dimensions=1024)

        vector = client.embed_text("provider outage incident")

        self.assertEqual(1024, len(vector))
        validate_embedding_dimension(vector, 1024, client.model_name)

    def test_role_prefix_helpers_are_idempotent(self) -> None:
        self.assertEqual("query: outage", build_query_text("outage"))
        self.assertEqual("query: outage", build_query_text("query: outage"))
        self.assertEqual("passage: incident report", build_document_text("incident report"))
        self.assertEqual("passage: incident report", build_document_text("passage: incident report"))


if __name__ == "__main__":
    unittest.main()
