import unittest
import mymod
class TestMyCode(unittest.TestCase):
    def test_double(self):
        x = 4
        expected = 2*x
        computed = mymod. double(x)
        self. assertEqual(expected, computed)
if __name__ == '__main__':
    unittest. main()