import unittest
from testsuite import TestSAT
from solution import sat

suite  = unittest.defaultTestLoader.loadTestsFromTestCase(TestSAT)
runner = unittest.TextTestRunner(resultclass=unittest.result.TestResult, verbosity=0)
result = runner.run(suite)
print(f"{result.testsRun},{len(result.failures)},{len(result.errors)}")
