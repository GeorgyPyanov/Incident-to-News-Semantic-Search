from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark_real import (
    INDEX_NAME,
    get_index_metadata,
    latency_statistics,
    parse_args,
    run_real_benchmark,
)


CPU_ONLY_HARDWARE = {
    "system": "TestOS",
    "release": "1.0",
    "machine": "test-machine",
    "python_version": "3.test",
    "cpu": {"model": "Test CPU", "logical_cores": 8, "physical_cores": 4},
    "total_ram_bytes": 16 * 1024**3,
    "gpu": {"model": "No GPU", "vram_bytes": None},
}


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.row: object = None
        self.rows: list[object] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: dict | None = None) -> None:
        self.statements.append(query)
        self.rows = []
        if "count(*) FILTER" in query:
            self.row = {
                "embedded_documents": 211,
                "total_documents": 250,
                "vector_dimension": 1024,
                "embedding_models": ["hashing-vectorizer-1024"],
            }
        elif "pg_relation_size" in query:
            self.row = {
                "index_name": INDEX_NAME,
                "index_type": "hnsw",
                "size_on_disk_bytes": 123456,
                "total_size_on_disk_bytes": 131072,
                "definition": (
                    "CREATE INDEX ix_structured_events_embedding ON public.structured_events "
                    "USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64') "
                    "WHERE (embedding IS NOT NULL)"
                ),
                "reloptions": ["m=16", "ef_construction=64"],
            }
        elif "benchmark_document_sample" in query:
            self.row = None
            self.rows = [
                {"id": "d1", "title": "First document"},
                {"id": "d2", "title": "Second document"},
            ]
        elif query.startswith("EXPLAIN"):
            self.row = {
                "QUERY PLAN": [
                    {
                        "Plan": {
                            "Node Type": "Limit",
                            "Plans": [
                                {
                                    "Node Type": "Index Scan",
                                    "Index Name": INDEX_NAME,
                                    "Relation Name": "structured_events",
                                }
                            ],
                        },
                        "Planning Time": 0.1,
                        "Execution Time": 0.2,
                    }
                ]
            }
        else:
            self.row = None

    def fetchone(self) -> object:
        return self.row

    def fetchall(self) -> list[object]:
        return self.rows


class FakeConnection:
    def __init__(self) -> None:
        self.fake_cursor = FakeCursor()
        self.commits = 0

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    def commit(self) -> None:
        self.commits += 1


class RealBenchmarkTests(unittest.TestCase):
    def test_result_schema_and_cpu_only_hardware(self) -> None:
        results, _ = self._run()

        self.assertEqual("real_postgresql_pgvector", results["benchmark_type"])
        self.assertEqual("No GPU", results["platform"]["gpu"]["model"])
        self.assertEqual(211, results["embedding"]["embedded_documents"])
        self.assertEqual(1024, results["embedding"]["vector_dimension"])
        self.assertEqual(1, results["embedding"]["batch_size"])
        self.assertEqual("hnsw", results["index"]["type"])
        self.assertIsNone(results["index"]["build_time_seconds"])
        self.assertTrue(results["explain_analyze"]["hnsw_index_used"])
        for category in (
            "query_embedding_latency_ms",
            "database_index_search_latency_ms",
            "end_to_end_retrieval_latency_ms",
        ):
            self.assertEqual(
                {"mean", "median", "p95", "min", "max"},
                set(results["search"][category]),
            )

    def test_latency_calculations(self) -> None:
        statistics = latency_statistics([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(22.0, statistics["mean"])
        self.assertEqual(3.0, statistics["median"])
        self.assertAlmostEqual(80.8, statistics["p95"])
        self.assertEqual(1.0, statistics["min"])
        self.assertEqual(100.0, statistics["max"])

    def test_saves_to_separate_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "benchmark_real_results.json"
            results, _ = self._run(output=output, save=True)
            self.assertTrue(output.exists())
            self.assertEqual(results, json.loads(output.read_text(encoding="utf-8")))

    def test_database_index_size_comes_from_metadata_query(self) -> None:
        cursor = FakeCursor()
        metadata = get_index_metadata(cursor)
        self.assertEqual(123456, metadata["size_on_disk_bytes"])
        self.assertTrue(any("pg_relation_size" in statement for statement in cursor.statements))

    def test_default_run_is_read_only_and_does_not_mutate(self) -> None:
        def forbidden_generator(limit: int) -> int:
            raise AssertionError(f"embedding generation must not run (limit={limit})")

        results, connection = self._run(embedding_generator=forbidden_generator)
        statements = connection.fake_cursor.statements
        self.assertTrue(results["database"]["read_only_run"])
        self.assertIn("SET TRANSACTION READ ONLY", statements)
        self.assertFalse(any("REINDEX" in statement.upper() for statement in statements))
        self.assertFalse(results["embedding"]["requested"])
        self.assertFalse(results["index"]["rebuild_requested"])
        self.assertNotIn("document_embedding_benchmark", results)

    def test_mutations_require_explicit_flags(self) -> None:
        args = parse_args([])
        self.assertFalse(args.generate_embeddings)
        self.assertFalse(args.rebuild_index)
        self.assertFalse(args.benchmark_document_embeddings)
        self.assertEqual(100, args.embedding_sample_size)

        generated_limits: list[int] = []
        results, connection = self._run(
            generate_embeddings=True,
            rebuild_index=True,
            embedding_generator=lambda limit: generated_limits.append(limit) or 2,
        )
        self.assertEqual([1000], generated_limits)
        self.assertTrue(any("REINDEX" in statement.upper() for statement in connection.fake_cursor.statements))
        self.assertEqual(1, connection.commits)
        self.assertFalse(results["database"]["read_only_run"])
        self.assertEqual(2, results["embedding"]["documents_generated"])
        self.assertIsNotNone(results["index"]["build_time_seconds"])

    def test_document_embedding_benchmark_excludes_warmup_and_never_writes(self) -> None:
        calls: list[str] = []

        def embedder(row: dict, dimension: int) -> list[float]:
            calls.append(row["id"])
            return [0.0] * dimension

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark_real_results.json"
            results, connection = self._run(
                output=output,
                save=True,
                benchmark_document_embeddings=True,
                embedding_sample_size=2,
                document_embedder=embedder,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        benchmark = results["document_embedding_benchmark"]
        self.assertEqual(["d1", "d1", "d2"], calls)
        self.assertEqual(benchmark, saved["document_embedding_benchmark"])
        self.assertEqual("hashing-vectorizer-1024", benchmark["model"])
        self.assertEqual(2, benchmark["sample_size"])
        self.assertEqual(1, benchmark["warmup_documents"])
        self.assertAlmostEqual(0.005, benchmark["total_generation_time_seconds"])
        self.assertEqual(
            {"mean", "median", "p95", "min", "max"},
            set(benchmark["per_document_latency_ms"]),
        )
        self.assertAlmostEqual(1.0, benchmark["per_document_latency_ms"]["mean"])
        self.assertAlmostEqual(400.0, benchmark["throughput_documents_per_second"])
        self.assertFalse(benchmark["database_writes_performed"])
        self.assertFalse(benchmark["embeddings_persisted"])
        self.assertFalse(benchmark["index_modified"])
        statements = "\n".join(connection.fake_cursor.statements).upper()
        for forbidden in ("INSERT ", "UPDATE ", "DELETE ", "REINDEX "):
            self.assertNotIn(forbidden, statements)
        self.assertIn("SET TRANSACTION READ ONLY", connection.fake_cursor.statements)

    def test_document_benchmark_rejects_mutation_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._run(
                benchmark_document_embeddings=True,
                generate_embeddings=True,
            )

    def _run(
        self,
        *,
        output: Path | None = None,
        save: bool = False,
        embedding_generator=None,
        generate_embeddings: bool = False,
        rebuild_index: bool = False,
        benchmark_document_embeddings: bool = False,
        embedding_sample_size: int = 100,
        document_embedder=None,
    ) -> tuple[dict, FakeConnection]:
        connection = FakeConnection()
        with tempfile.TemporaryDirectory() as temp_dir:
            validation = Path(temp_dir) / "validation.json"
            validation.write_text(
                json.dumps(
                    {
                        "examples": [
                            {"query": {"message": "Twilio SMS delivery incident"}},
                            {"query": {"message": "GitHub service degradation"}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            results = run_real_benchmark(
                validation_path=validation,
                output_path=output or Path(temp_dir) / "unused.json",
                searches=100,
                warmups=1,
                connection_factory=lambda: connection,
                query_embedder=lambda query, dimension: "[0,0]",
                embedding_generator=embedding_generator,
                generate_embeddings=generate_embeddings,
                rebuild_index=rebuild_index,
                benchmark_document_embeddings=benchmark_document_embeddings,
                embedding_sample_size=embedding_sample_size,
                document_embedder=document_embedder,
                search_sql="SELECT real_pgvector_search",
                hardware=CPU_ONLY_HARDWARE,
                timestamp="2026-07-14T00:00:00+00:00",
                clock=StepClock(),
                save=save,
            )
        return results, connection


if __name__ == "__main__":
    unittest.main()
