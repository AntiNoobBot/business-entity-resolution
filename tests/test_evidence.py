import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_final_recall_matches_retained_counts(self):
        report = (ROOT / "reports/historical/blocking_v11_finetuned_india_report.txt").read_text()
        recall = float(re.search(r"v11 candidate recall: ([\d.]+)%", report).group(1))
        self.assertAlmostEqual(recall, (76767 - 810) / 76767 * 100, places=6)

    def test_retrieval_gains_and_budgets_add_up(self):
        stages = [
            (8, "blocking_v8_address_char_report.txt", "retrieval-added"),
            (9, "blocking_v9_address_name_char_report.txt", "name-retrieval-added"),
            (10, "blocking_v10_multilingual_report.txt", "multilingual-added"),
            (11, "blocking_v11_finetuned_india_report.txt", "fine-tuned-added"),
        ]
        previous = 2530505
        recovered = 65472
        for version, filename, prefix in stages:
            report = (ROOT / "reports/historical" / filename).read_text()
            def count(pattern):
                return int(re.search(pattern + r": ([\d,]+)", report).group(1).replace(",", ""))
            current = count(f"v{version} total candidates")
            self.assertEqual(current - previous, count(prefix + " candidates"))
            recovered += count(prefix + " true links")
            recall = float(re.search(f"v{version} candidate recall: ([\\d.]+)%", report).group(1))
            self.assertAlmostEqual(recovered / 76767 * 100, recall, places=6)
            previous = current
        self.assertEqual(76767 - recovered, 810)

    def test_historical_feature_schema_is_35_inputs(self):
        path = ROOT / "experiments/hackathon/19_train_lgbm_v2.py"
        tree = ast.parse(path.read_text())
        features = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "FEATURES" for target in node.targets))
        self.assertEqual(len(features) + 2, 35)
        self.assertEqual(len(features), len(set(features)))
