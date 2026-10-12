"""Core invariants without generated results or a native build."""

import unittest

import numpy as np

from mapchain.evaluation.metrics import fast_cones, ordered_cone_metrics
from mapchain.volumetric.hilbert import hilbert_nd


class HilbertTests(unittest.TestCase):
    def test_complete_adjacent_nested_traversals(self):
        for dimensions in (2, 3):
            previous = None
            for order in range(1, 5):
                with self.subTest(dimensions=dimensions, order=order):
                    side = 2**order
                    points = hilbert_nd(order, dimensions)
                    cells = np.rint(points * side - 0.5).astype(int)
                    self.assertEqual(points.shape, (side**dimensions, dimensions))
                    self.assertEqual(len(np.unique(cells, axis=0)), len(cells))
                    self.assertTrue(np.all((cells >= 0) & (cells < side)))
                    self.assertTrue(np.all(np.abs(np.diff(cells, axis=0)).sum(axis=1) == 1))
                    if previous is not None:
                        parents = (cells >> 1).reshape(-1, 2**dimensions, dimensions)
                        np.testing.assert_array_equal(parents[:, 0], previous)
                        self.assertTrue(np.all(parents == parents[:, :1]))
                    previous = cells

    def test_invalid_grid(self):
        for order, dimensions in ((0, 3), (-1, 2), (2, 1)):
            with self.subTest(order=order, dimensions=dimensions):
                with self.assertRaises(ValueError):
                    hilbert_nd(order, dimensions)


class ConeTests(unittest.TestCase):
    def test_spatial_pruning_matches_exhaustive_checks(self):
        rng = np.random.default_rng(7)
        for points in [rng.random((180, 3)), hilbert_nd(2, 3)]:
            reference = ordered_cone_metrics(points, 0.12)
            for batch_size in (1, 16, 2048):
                with self.subTest(count=len(points), batch_size=batch_size):
                    self.assertEqual(fast_cones(points, 0.12, batch_size=batch_size), reference)

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            fast_cones(np.zeros((2, 3)), 0.1, batch_size=0)


if __name__ == "__main__":
    unittest.main()
