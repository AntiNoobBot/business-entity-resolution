import math
import unittest

from er_lab.selection import ScoredPair, select_pairs


class SelectionTests(unittest.TestCase):
    def test_multi_match_and_singleton(self):
        result = select_pairs(["S1-a", "S1-b"], [
            ScoredPair("S1-a", "S2-x", .9), ScoredPair("S1-a", "S3-y", .8),
        ])
        self.assertEqual(result, {"S1-a": {"S2-x", "S3-y"}, "S1-b": set()})

    def test_owner_tie_is_order_independent(self):
        pairs = [ScoredPair("S1-b", "S2-x", .8), ScoredPair("S1-a", "S2-x", .8)]
        expected = {"S1-a": {"S2-x"}, "S1-b": set()}
        self.assertEqual(select_pairs(expected, pairs), expected)
        self.assertEqual(select_pairs(expected, reversed(pairs)), expected)

    def test_duplicate_pair_uses_max_score(self):
        pairs = [ScoredPair("S1-a", "S2-x", .9), ScoredPair("S1-a", "S2-x", .1)]
        self.assertEqual(select_pairs(["S1-a"], pairs)["S1-a"], {"S2-x"})

    def test_ownership_can_be_disabled(self):
        pairs = [ScoredPair("S1-a", "S2-x", .9), ScoredPair("S1-b", "S2-x", .8)]
        result = select_pairs(["S1-a", "S1-b"], pairs, one_owner=False)
        self.assertEqual(result["S1-a"], result["S1-b"])

    def test_threshold_is_inclusive(self):
        self.assertEqual(select_pairs(["S1-a"], [ScoredPair("S1-a", "S2-x", .65)])["S1-a"], {"S2-x"})

    def test_bad_inputs_fail_before_selection(self):
        for score in (math.nan, math.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                select_pairs(["S1-a"], [ScoredPair("S1-a", "S2-x", score)])
        for pair in (ScoredPair("S1-unknown", "S2-x", .2), ScoredPair("S1-a", "S1-x", .2)):
            with self.assertRaises(ValueError):
                select_pairs(["S1-a"], [pair])
        with self.assertRaises(ValueError):
            select_pairs(["S1-a", "S1-a"], [])

