import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from er_lab.io import read_pair_lists, write_pair_lists


class FileTests(unittest.TestCase):
    def test_round_trip_with_empty_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.tsv"
            values = {"S1-a": {"S2-x", "S3-y"}, "S1-b": set()}
            write_pair_lists(path, values)
            self.assertEqual(read_pair_lists(path), values)
            with self.assertRaises(FileExistsError):
                write_pair_lists(path, values)

    def test_malformed_input(self):
        header = "source1_entity_id\tmatched_entity_ids\n"
        invalid = [
            "wrong\theader\n",
            header + "S1-a\tS2-x,S2-x\n",
            header + "S1-a\tS2-x\nS1-a\t\n",
            header + "S1-a\t S2-x\n",
            header + "S1-a\tS1-x\n",
            header + "S1-a\n",
            header + "S1-a\tS2-x\textra\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.tsv"
            for text in invalid:
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_pair_lists(path)

    def test_cli_demo_and_scoring_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            first = subprocess.run(
                [sys.executable, "-m", "er_lab", "demo", "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            second = subprocess.run(
                [sys.executable, "-m", "er_lab", "score", "--truth", str(output / "truth.tsv"),
                 "--predictions", str(output / "matching_results.tsv"),
                 "--candidates", str(output / "candidate_pairs.tsv")],
                check=True, capture_output=True, text=True,
            )
            demonstrated = json.loads(first.stdout)
            demonstrated.pop("scope")
            self.assertEqual(demonstrated, json.loads(second.stdout))

