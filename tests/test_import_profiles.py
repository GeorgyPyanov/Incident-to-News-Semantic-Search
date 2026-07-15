from __future__ import annotations

import unittest
from types import SimpleNamespace

from data.import_datasets import apply_profile


class ImportProfileTests(unittest.TestCase):
    def test_large_profile_sets_large_scale_import_limits(self) -> None:
        args = SimpleNamespace(profile="large", minimum_total=50_000)

        minimum_total = apply_profile(args)

        self.assertEqual(500_000, minimum_total)
        self.assertEqual(500_000, args.minimum_total)
        self.assertEqual(400_000, args.gdeltv2_limit)
        self.assertEqual(100_000, args.gharchive_log_limit)
        self.assertEqual(5_000, args.gharchive_news_limit)

    def test_legacy_project_500k_profile_remains_supported(self) -> None:
        args = SimpleNamespace(profile="project_500k", minimum_total=50_000)

        minimum_total = apply_profile(args)

        self.assertEqual(500_000, minimum_total)
        self.assertEqual(400_000, args.gdeltv2_limit)


if __name__ == "__main__":
    unittest.main()
