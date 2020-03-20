import unittest

from metrics.recommendations import *


class RecommendationsMetricsTestCase(unittest.TestCase):
    def test_reciprocal_rank(self):
        self.assertEqual(
            1,
            reciprocal_rank([1, 2, 3], [1, 0, 0])
        )
        self.assertEqual(
            0,
            reciprocal_rank([1, 2, 3], [0, 0, 0])
        )


if __name__ == '__main__':
    unittest.main()
