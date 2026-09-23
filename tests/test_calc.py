import unittest
from calc import apply_tax

class TestTaxCalculation(unittest.TestCase):
    def test_apply_tax(self):
        self.assertEqual(apply_tax(100.0, 25), 125.0)

if __name__ == '__main__':
    unittest.main()