import copy
import json
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goalpost.markets import bridge as h


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.export = self.root / "export"
        shutil.copytree(h.HERE / "example/export", self.export)

    def edit(self, action):
        path = self.export / "report.json"
        report = json.loads(path.read_text())
        action(report)
        path.write_text(json.dumps(report))

    def test_demo_exact_known_answers_and_no_network(self):
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        ):
            result = h.run(None, self.root / "demo", demo=True)
        rows = json.loads(
            (Path(result["output"]) / "comparison/current-market-sizing.json").read_text()
        )
        self.assertEqual(len(rows), 6)
        home = next(r for r in rows if r["market"] == "h2h" and r["participant"] == "Example Home")
        self.assertAlmostEqual(home["p_win"], 0.7)
        self.assertAlmostEqual(home["p_push_or_regulation_tie"], 0.1)
        self.assertAlmostEqual(home["hypothetical_ev_pct"], 15)
        self.assertAlmostEqual(home["quarter_kelly_pct"], 100 / 12)
        self.assertAlmostEqual(home["p_win_all_attempts_lower"], 0.7 * 100 / 120)
        self.assertIsNone(home["actual_contract_ev"])
        self.assertIsNone(home["actual_contract_kelly"])
        self.assertFalse(home["betting_admitted"])
        self.assertNotIn("hypothetical_quarter_kelly_dollars", home)
        self.assertEqual(result["messages_sent"], 0)

    def test_demo_cannot_mix_real_inputs(self):
        with self.assertRaisesRegex(ValueError, "demo_cannot_use_live"):
            h.run(self.export, self.root / "out", demo=True)

    def test_scope_mismatch_rejected(self):
        self.edit(lambda r: r.update(regulation_only=False))
        with self.assertRaisesRegex(ValueError, "contract_scope"):
            h.validate_export(self.export)

    def test_home_away_unknown_rejected(self):
        self.edit(lambda r: r["results"][0]["game"].update(home="UNKNOWN"))
        with self.assertRaisesRegex(ValueError, "home_away"):
            h.validate_export(self.export)

    def test_directory_traversal_rejected(self):
        self.edit(lambda r: r["results"][0]["game"].update(game_id="../escape"))
        with self.assertRaisesRegex(ValueError, "unsafe_game_directory"):
            h.validate_export(self.export)

    def test_duplicate_game_rejected(self):
        self.edit(lambda r: r["results"].append(copy.deepcopy(r["results"][0])))
        with self.assertRaisesRegex(ValueError, "duplicate_game_id"):
            h.validate_export(self.export)

    def test_probability_counts_mismatch_rejected(self):
        path = self.export / "DEMO_AWAY_HOME/total-pmf.csv"
        path.write_text(path.read_text().replace("37,20,0.2", "37,20,0.3"))
        with self.assertRaises(ValueError):
            h.validate_export(self.export)

    def test_missing_margin_cannot_be_replaced_by_total(self):
        (self.export / "DEMO_AWAY_HOME/home_minus_away-pmf.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            h.validate_export(self.export)

    def test_validate_only_no_prices_or_network(self):
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        ):
            result = h.run(self.export, self.root / "validate", validate_only=True)
        self.assertEqual(result["status"], "validated_input_schema_and_pmf_counts")
        self.assertFalse((self.root / "validate/comparison").exists())

    def test_bad_dataset_hash_cannot_price(self):
        with self.assertRaisesRegex(ValueError, "missing_export_dataset_hash"):
            h.verify_dataset(self.root, {})
        data = self.root / "data"
        data.mkdir()
        for name in ("transition_set.json", "nfl_games.parquet", "parsed_segments.parquet"):
            (data / name).write_text("sample")
        with self.assertRaisesRegex(ValueError, "does_not_match"):
            h.verify_dataset(data, {"dataset_hashes": {"transition_set.json": "0" * 64}})

    def test_synthetic_export_never_live(self):
        with self.assertRaisesRegex(ValueError, "synthetic_export_requires_demo"):
            h.run(self.export, self.root / "out", data=self.root, database=self.root / "db")

    def test_freshness_block_never_produces_comparison(self):
        self.edit(lambda r: r.pop("synthetic_demo"))
        with (
            patch.object(h, "verify_dataset"),
            patch.object(h.history, "check") as check,
            patch.object(h.comparison, "compare") as compare,
        ):
            check.side_effect = ValueError("Model data freshness blocked")
            with self.assertRaisesRegex(ValueError, "freshness blocked"):
                h.run(self.export, self.root / "blocked", data=self.root, database=self.root / "db")
            compare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
