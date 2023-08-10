import sys
import unittest

sys.path.append('../xia_eeg_toolkit')


class MyTestCase(unittest.TestCase):
    def test_(self):
        self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()
