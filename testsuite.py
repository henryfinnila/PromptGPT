import unittest
import random
from solution import sat

class TestSAT(unittest.TestCase):
    # 1) Hand-crafted edge cases
    test_cases = [
        {"name": "Test 1: Empty formula",        "clauses": []},
        {"name": "Test 2: Empty clause",         "clauses": [[]]},
        {"name": "Test 3: Single positive var",   "clauses": [[1]]},
        {"name": "Test 4: Single negative var",   "clauses": [[-1]]},
        {"name": "Test 5: Simple AND sat",        "clauses": [[1], [2]]},
        {"name": "Test 6: Simple OR sat",         "clauses": [[1, 2]]},
        {"name": "Test 7: Contradiction unsat",   "clauses": [[1], [-1]]},
        {"name": "Test 8: Tautology sat",         "clauses": [[1, -1]]},
        {"name": "Test 9: Two clauses unsat",     "clauses": [[1, 2], [-1, -2]]},
        {"name": "Test 10: Three vars mixed",     "clauses": [[1, -2], [2, 3], [-3]]},
    ]
    # 2) Add randomly generated cases to reach 100 tests
    random.seed(42)
    for i in range(11, 101):
        n = random.randint(1, 6)
        num_clauses = random.randint(1, 1 << n)
        clauses = []
        for _ in range(num_clauses):
            size = random.randint(1, n)
            vars_ = random.sample(range(1, n+1), size)
            clause = [v if random.choice([True, False]) else -v for v in vars_]
            clauses.append(clause)
        test_cases.append({"name": f"Random Case {i}", "clauses": clauses})

    def brute_sat(self, clauses):
        n = max((abs(lit) for c in clauses for lit in c), default=0)
        for mask in range(1 << n):
            ok = True
            for clause in clauses:
                if not any(((lit > 0) == bool((mask >> (abs(lit)-1)) & 1)) for lit in clause):
                    ok = False
                    break
            if ok:
                return True
        return False

    def test_all_cases(self):
        for tc in self.test_cases:
            with self.subTest(tc["name"]):
                expected = self.brute_sat(tc["clauses"])
                result = sat(tc["clauses"])
                self.assertEqual(
                    result,
                    expected,
                    msg=f"{tc['name']} failed: clauses={tc['clauses']}, expected {expected}, got {result}"
                )

if __name__ == '__main__':
    unittest.main()
