import math
import unittest

from er_lab.demo import demo
from er_lab.metrics import entity_fbeta, evaluate


class MetricTests(unittest.TestCase):
    def test_empty_set_convention(self):
        self.assertEqual(entity_fbeta(set(), set()), 1)
        self.assertEqual(entity_fbeta(set(), {"x"}), 0)
        self.assertEqual(entity_fbeta({"x"}, set()), 0)

    def test_precision_penalty(self):
        self.assertAlmostEqual(entity_fbeta({"a", "b"}, {"a", "b", "wrong"}), 5 / 7)
        self.assertGreater(entity_fbeta({"a", "b"}, {"a"}), entity_fbeta({"a"}, {"a", "wrong"}))

    def test_macro_is_not_pooled_recall(self):
        truth = {"a": set(map(str, range(10))), "b": {"target"}}
        pred = {"a": truth["a"], "b": {"wrong"}}
        result = evaluate(truth, pred)
        self.assertEqual(result["macro_f0_5"], 0.5)
        self.assertAlmostEqual(result["micro_recall"], 10 / 11)

    def test_missing_and_extra_entities_are_rejected(self):
        for pred in [{}, {"a": set(), "extra": set()}]:
            with self.assertRaises(ValueError):
                evaluate({"a": set()}, pred)

    def test_predictions_must_be_candidates(self):
        with self.assertRaises(ValueError):
            evaluate({"a": {"x"}}, {"a": {"x"}}, {"a": set()})

    def test_oracle_does_not_equal_actual_model_score(self):
        _, _, _, result = demo()
        self.assertAlmostEqual(result["macro_f0_5"], 61 / 72)
        self.assertEqual(result["candidate_recall"], 0.75)
        self.assertAlmostEqual(result["oracle_macro_f0_5"], 23 / 24)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertGreater(result["oracle_macro_f0_5"], result["macro_f0_5"])

    def test_no_true_links_has_undefined_link_recall(self):
        result = evaluate({"a": set()}, {"a": set()}, {"a": set()})
        self.assertEqual(result["macro_f0_5"], 1)
        self.assertIsNone(result["candidate_recall"])
        self.assertIsNone(result["micro_precision"])

    def test_empty_evaluation_and_invalid_beta_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({}, {})
        for beta in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                entity_fbeta({"a"}, {"a"}, beta)

