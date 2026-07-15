from __future__ import annotations

import os
import unittest

from evaluation.compare_iterations import IterationConfig, _delta, temporary_environment


class CompareIterationsTests(unittest.TestCase):
    def test_iteration_config_exports_environment(self) -> None:
        config = IterationConfig(
            name="candidate",
            embedding_backend="sentence-transformer",
            embedding_model="intfloat/e5-small-v2",
            embedding_quantization="dynamic",
            fusion_mode="normalized_sum",
            refresh_embeddings=True,
        )

        self.assertEqual(
            {
                "EMBEDDING_BACKEND": "sentence-transformer",
                "EMBEDDING_MODEL": "intfloat/e5-small-v2",
                "EMBEDDING_QUANTIZATION": "dynamic",
                "RETRIEVAL_FUSION_MODE": "normalized_sum",
            },
            config.env(),
        )

    def test_temporary_environment_restores_previous_values(self) -> None:
        previous_quantization = os.environ.get("EMBEDDING_QUANTIZATION")
        previous_fusion = os.environ.pop("RETRIEVAL_FUSION_MODE", None)
        os.environ["EMBEDDING_QUANTIZATION"] = "none"
        try:
            with temporary_environment({"EMBEDDING_QUANTIZATION": "dynamic", "RETRIEVAL_FUSION_MODE": "normalized_sum"}):
                self.assertEqual("dynamic", os.environ["EMBEDDING_QUANTIZATION"])
                self.assertEqual("normalized_sum", os.environ["RETRIEVAL_FUSION_MODE"])

            self.assertEqual("none", os.environ["EMBEDDING_QUANTIZATION"])
            self.assertNotIn("RETRIEVAL_FUSION_MODE", os.environ)
        finally:
            if previous_quantization is None:
                os.environ.pop("EMBEDDING_QUANTIZATION", None)
            else:
                os.environ["EMBEDDING_QUANTIZATION"] = previous_quantization
            if previous_fusion is not None:
                os.environ["RETRIEVAL_FUSION_MODE"] = previous_fusion

    def test_delta_reports_absolute_and_relative_changes(self) -> None:
        baseline = _result(0.8, 12.0)
        candidate = _result(0.84, 9.0)

        delta = _delta(baseline, candidate)

        self.assertAlmostEqual(0.8, delta["qrels_hybrid_ndcg_at_k"]["baseline"])
        self.assertAlmostEqual(0.84, delta["qrels_hybrid_ndcg_at_k"]["candidate"])
        self.assertAlmostEqual(0.04, delta["qrels_hybrid_ndcg_at_k"]["absolute"])
        self.assertAlmostEqual(0.05, delta["qrels_hybrid_ndcg_at_k"]["relative"])
        self.assertAlmostEqual(-3.0, delta["document_embedding_mean_ms"]["absolute"])


def _result(ndcg: float, embedding_ms: float) -> dict:
    return {
        "qrels_validation": {"modes": {"hybrid": {"ndcg_at_k": ndcg, "mrr_at_k": ndcg}}},
        "linked_validation": {"modes": {"hybrid": {"ndcg_at_k": ndcg}}},
        "document_embedding_benchmark": {"per_document_latency_ms": {"mean": embedding_ms}},
        "search": {
            "database_index_search_latency_ms": {"mean": 2.0},
            "end_to_end_retrieval_latency_ms": {"mean": 14.0},
        },
    }


if __name__ == "__main__":
    unittest.main()
