import unittest
from collections import Counter
from functools import lru_cache

# Import the solution function.
# It is assumed that the implementation of max_domino_chain is in a file named solution.py.
from solution import max_domino_chain


class TestMaxDominoChain(unittest.TestCase):
    def canonical(self, domino):
        """Return the canonical (sorted) representation of a domino piece.
        This is used so that a piece (a, b) and its flipped version (b, a) are considered the same."""
        return tuple(sorted(domino))

    def is_valid_chain(self, dominoes, chain):
        """
        Checks that:
          a) the chain is connected (i.e. chain[i][1] == chain[i+1][0] for every i) and
          b) each domino used (in either orientation) is no more frequent than in the original list.
        """
        # For an empty chain, we accept it only if no dominoes were given.
        if not chain:
            return len(dominoes) == 0

        # Check connectivity.
        for i in range(len(chain) - 1):
            if chain[i][1] != chain[i + 1][0]:
                return False

        # Count frequency (by canonical form) in the input and the output chain.
        input_count = Counter(self.canonical(d) for d in dominoes)
        chain_count = Counter(self.canonical(d) for d in chain)
        for domino, count in chain_count.items():
            if count > input_count.get(domino, 0):
                return False
        return True

    def compute_max_chain_length(self, dominoes):
        """
        Uses backtracking with memoization to determine the maximum number of dominoes that can be
        used in a valid chain given the input dominoes. Domino pieces may be flipped.
        """
        n = len(dominoes)

        @lru_cache(maxsize=None)
        def dfs(current, used):
            best = 0
            for i in range(n):
                # Check if domino i has not been used.
                if not (used & (1 << i)):
                    a, b = dominoes[i]
                    # Option 1: use domino as given if a matches the current chain end (or if starting fresh).
                    if current is None or a == current:
                        best = max(best, 1 + dfs(b, used | (1 << i)))
                    # Option 2: flip the domino if b matches the current chain end.
                    if current is None or b == current:
                        # Only consider the flip separately if a and b are different to avoid duplicate work.
                        if a != b:
                            best = max(best, 1 + dfs(a, used | (1 << i)))
            return best

        return dfs(None, 0)

    def test_all_cases(self):
        # Define a list of 50 test cases.
        # Each dictionary contains:
        #  - a short descriptive name,
        #  - the input list "dominoes" (each domino as a tuple),
        #  - and "expected", the maximum number of dominoes that can be chained.
        test_cases = [
            {"name": "Test 1: Empty input", "dominoes": [], "expected": 0},
            {"name": "Test 2: Single domino", "dominoes": [(1, 2)], "expected": 1},
            {"name": "Test 3: Two dominoes, directly connectable", "dominoes": [(1, 2), (2, 3)], "expected": 2},
            {"name": "Test 4: Two dominoes, second requires flip", "dominoes": [(2, 1), (2, 3)], "expected": 2},
            {"name": "Test 5: Two dominoes, not connectable", "dominoes": [(1, 2), (3, 4)], "expected": 1},
            {"name": "Test 6: Three dominoes, direct chain", "dominoes": [(1, 2), (2, 3), (3, 4)], "expected": 3},
            {"name": "Test 7: Three dominoes, requiring flip", "dominoes": [(2, 1), (2, 3), (3, 1)], "expected": 3},
            {"name": "Test 8: Three dominoes, sample problem", "dominoes": [(1, 2), (3, 1), (2, 3)], "expected": 3},
            {"name": "Test 9: Four dominoes, one disconnected", "dominoes": [(1, 2), (2, 3), (3, 4), (5, 6)], "expected": 3},
            {"name": "Test 10: Four dominoes, fully chainable", "dominoes": [(1, 2), (2, 3), (3, 1), (1, 4)], "expected": 4},
            {"name": "Test 11: Five dominoes, full chain", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], "expected": 5},
            {"name": "Test 12: Five dominoes, one isolated", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 5), (7, 8)], "expected": 4},
            {"name": "Test 13: Three dominoes, chain with essential flip", "dominoes": [(2, 3), (4, 2), (3, 1)], "expected": 3},
            {"name": "Test 14: Three dominoes with duplicate domino", "dominoes": [(1, 2), (2, 3), (1, 2)], "expected": 3},
            {"name": "Test 15: Four dominoes with same numbers", "dominoes": [(1, 1), (1, 2), (2, 1), (2, 2)], "expected": 4},
            {"name": "Test 16: Three dominoes, partly unchainable", "dominoes": [(3, 4), (4, 5), (1, 2)], "expected": 2},
            {"name": "Test 17: Five dominoes, multiple possibilities", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 1), (2, 2)], "expected": 5},
            {"name": "Test 18: Four identical dominoes", "dominoes": [(1, 2), (1, 2), (1, 2), (1, 2)], "expected": 4},
            {"name": "Test 19: Three dominoes, flip in the middle", "dominoes": [(3, 1), (2, 3), (1, 4)], "expected": 3},
            {"name": "Test 20: Four dominoes, one isolated piece", "dominoes": [(1, 2), (2, 3), (3, 4), (5, 7)], "expected": 3},
            {"name": "Test 21: Five dominoes forming a circle", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)], "expected": 5},
            {"name": "Test 22: Four dominoes with branch", "dominoes": [(1, 2), (2, 3), (3, 4), (2, 5)], "expected": 3},
            {"name": "Test 23: Four dominoes out of order", "dominoes": [(4, 5), (2, 3), (1, 2), (3, 4)], "expected": 4},
            {"name": "Test 24: Four dominoes, multiple matches", "dominoes": [(1, 2), (2, 1), (2, 3), (3, 2)], "expected": 4},
            {"name": "Test 25: Three dominoes, no connections", "dominoes": [(1, 2), (3, 4), (5, 6)], "expected": 1},
            {"name": "Test 26: Four dominoes, potential flip error", "dominoes": [(2, 3), (3, 4), (4, 2), (2, 1)], "expected": 4},
            {"name": "Test 27: Five dominoes, one branch doesn't connect", "dominoes": [(1, 2), (2, 3), (5, 6), (3, 4), (7, 8)], "expected": 3},
            {"name": "Test 28: Six dominoes, full chain", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], "expected": 6},
            {"name": "Test 29: Five dominoes, cyclic chain", "dominoes": [(1, 3), (3, 5), (5, 7), (7, 9), (9, 1)], "expected": 5},
            {"name": "Test 30: Three dominoes, missing connection", "dominoes": [(1, 3), (4, 5), (3, 2)], "expected": 2},
            {"name": "Test 31: Six dominoes, long chain", "dominoes": [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 2)], "expected": 6},
            {"name": "Test 32: Four dominoes needing inversion", "dominoes": [(4, 2), (6, 4), (8, 6), (10, 8)], "expected": 4},
            {"name": "Test 33: Four dominoes, repeated connections", "dominoes": [(1, 2), (2, 3), (3, 1), (1, 3)], "expected": 4},
            {"name": "Test 34: Six dominoes, extra piece", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 9)], "expected": 5},
            {"name": "Test 35: Five dominoes, branch extension", "dominoes": [(3, 4), (4, 5), (5, 6), (6, 3), (3, 7)], "expected": 5},
            {"name": "Test 36: Four dominoes, chainable after flip", "dominoes": [(2, 3), (3, 4), (4, 5), (5, 2)], "expected": 4},
            {"name": "Test 37: Five dominoes, alternating pattern", "dominoes": [(1, 2), (2, 1), (1, 2), (2, 1), (1, 2)], "expected": 5},
            {"name": "Test 38: Six dominoes, maximum chain uses subset", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 5), (7, 8), (8, 9)], "expected": 4},
            {"name": "Test 39: Five dominoes, disconnected segments", "dominoes": [(1, 2), (2, 3), (4, 5), (5, 6), (6, 7)], "expected": 3},
            {"name": "Test 40: Four dominoes, overlapping numbers", "dominoes": [(1, 3), (3, 3), (3, 5), (5, 1)], "expected": 4},
            {"name": "Test 41: Five dominoes, unordered input", "dominoes": [(4, 6), (1, 3), (6, 2), (2, 1), (3, 4)], "expected": 5},
            {"name": "Test 42: Six dominoes, tricky case", "dominoes": [(1, 2), (2, 3), (3, 4), (4, 3), (3, 2), (2, 1)], "expected": 6},
            {"name": "Test 43: Three dominoes, identical doubles", "dominoes": [(2, 2), (2, 2), (2, 2)], "expected": 3},
            {"name": "Test 44: Four dominoes, duplicate mix", "dominoes": [(1, 2), (2, 3), (3, 1), (1, 2)], "expected": 4},
            {"name": "Test 45: Five dominoes, extra isolated", "dominoes": [(5, 6), (6, 7), (7, 8), (8, 9), (1, 2)], "expected": 4},
            {"name": "Test 46: Three dominoes, cycle", "dominoes": [(1, 2), (2, 3), (3, 1)], "expected": 3},
            {"name": "Test 47: Three dominoes, repeated number", "dominoes": [(1, 2), (2, 2), (2, 3)], "expected": 3},
            {"name": "Test 48: Five dominoes, reversal possibility", "dominoes": [(2, 3), (3, 4), (4, 2), (2, 5), (5, 6)], "expected": 5},
            {"name": "Test 49: Four dominoes, repeated pair arrangement", "dominoes": [(3, 4), (3, 4), (4, 5), (5, 3)], "expected": 4},
            {"name": "Test 50: Fifteen dominoes, maximum complexity",
             "dominoes": [(1, 2), (2, 2), (2, 3), (3, 3), (3, 4),
                          (4, 4), (4, 5), (5, 5), (5, 6), (6, 7),
                          (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
             "expected": 15},
        ]

        for tc in test_cases:
            with self.subTest(tc["name"]):
                dominoes = tc["dominoes"]
                expected = tc["expected"]
                # Use our backtracking solver to compute the max chain length.
                computed_expected = self.compute_max_chain_length(dominoes)
                # (This sanity-check ensures our expected value is consistent.)
                self.assertEqual(computed_expected, expected,
                                 msg=f"Internal max chain computation mismatch for {tc['name']}")
                # Get the chain returned by the student's solution.
                chain = max_domino_chain(dominoes)
                # Check that the domino chain is valid.
                self.assertTrue(self.is_valid_chain(dominoes, chain),
                                msg=f"Invalid chain for {tc['name']}: {chain}")
                # Verify that the chain uses the maximum number of pieces.
                self.assertEqual(len(chain), expected,
                                 msg=f"Chain length incorrect for {tc['name']}: expected {expected}, got {len(chain)}")

if __name__ == '__main__':
    unittest.main()
