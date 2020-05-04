import unittest

from metrics.recommendations import (
    reciprocal_rank, mean_reciprocal_rank, ndcg, discounted_cumulative_gain
)


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
        self.assertEqual(
            1 / 100,
            reciprocal_rank(
                [x for x in range(100)],
                [-1 for _ in range(99)] + [99]
            )
        )

    def test_mean_reciprocal_rank(self):
        self.assertEqual(
            1,
            mean_reciprocal_rank(
                [[1, 2, 3]],
                [[1, 0, 0]]
            )
        )
        self.assertEqual(
            1,
            mean_reciprocal_rank(
                [[1, 2, 3], [1, 2, 3]],
                [[1, 0, 0], [1, 0, 0]]
            )
        )
        self.assertEqual(
            1 / 3,
            mean_reciprocal_rank(
                [[1, 2, 3], [1, 2, 3]],
                [[0, 0, 1], [0, 0, 1]]
            )
        )
        self.assertEqual(
            1 / 2,
            mean_reciprocal_rank(
                [['a', 'b', 'c'], ['a', 'b', 'c']],
                [['d', 'b', 'g'], ['z', 'a', 'q']]
            )
        )

    def test_discounted_cumulative_gain(self):
        self.assertAlmostEqual(
            13.848,
            discounted_cumulative_gain(
                [3, 2, 3, 0, 1, 2]
            ),
            places=2
        )

    def test_ndcg(self):
        self.assertAlmostEqual(
            0.948,
            ndcg(
                [3, 3, 2, 2, 1, 0],
                [3, 2, 3, 0, 1, 2]
            ),
            places=2
        )


if __name__ == '__main__':
    unittest.main()
