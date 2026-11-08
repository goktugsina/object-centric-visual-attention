import unittest
import numpy as np
from scipy.optimize import check_grad
from attention.data import partition_masks, fixation_pixels, split_images
from attention.metrics import scores, entropy, bootstrap_mean_ci
from attention.model import objective, packed_objective, temperature_scale


class ScientificContracts(unittest.TestCase):
    def test_overlaps_and_background_are_disjoint(self):
        big = np.zeros((4, 5), bool); big[1:4, 1:5] = True
        small = np.zeros_like(big); small[2, 2] = True
        labels = partition_masks([big, small])
        self.assertEqual(labels[2, 2], 2)
        self.assertEqual(labels[1, 1], 1)
        self.assertEqual(labels[0, 0], 0)
        self.assertEqual(np.bincount(labels.ravel()).sum(), 20)

    def test_equal_area_overlap_uses_first_source_object(self):
        masks = np.ones((2, 2, 2), bool)
        np.testing.assert_array_equal(partition_masks(masks), np.ones((2, 2)))

    def test_nearest_policy_uses_only_covering_objects(self):
        a = np.zeros((5, 5), bool); a[:, :3] = True
        b = np.zeros((5, 5), bool); b[:, 2:] = True
        labels = partition_masks([a, b], policy='nearest')
        self.assertEqual(labels[2, 0], 1)
        self.assertEqual(labels[2, 4], 2)
        self.assertEqual(labels[2, 2], 1)

    def test_coordinates_and_outside_samples(self):
        x, y, valid = fixation_pixels([1, 2.5, 5, 0, 6, np.nan], [1, 2, 4, 2, 2, 2], 5, 4)
        np.testing.assert_array_equal(x, [0, 2, 4])
        np.testing.assert_array_equal(y, [0, 1, 3])
        self.assertEqual(valid.sum(), 3)

    def test_split_has_no_image_leakage_and_is_order_invariant(self):
        names = [f'{i:04}.jpg' for i in range(700)]
        split = split_images(names)
        self.assertEqual(split, split_images(names[::-1]))
        self.assertEqual([len(v) for v in split.values()], [420, 140, 140])
        self.assertEqual(len(set(sum(split.values(), []))), 700)

    def test_brier_is_expected_one_hot_score_not_distance_to_mean(self):
        q, p = np.array([.25, .75]), np.array([.4, .6])
        expected = sum(q[j] * np.sum((p-np.eye(2)[j])**2) for j in range(2))
        self.assertAlmostEqual(scores(q, p)['brier'], expected)
        self.assertAlmostEqual(scores(q, q)['brier'], 1-q@q)
        self.assertLess(scores(q, q)['nll_bits'], scores(q, p)['nll_bits'])

    def test_gradient_against_finite_differences(self):
        xs = [np.array([[1., 0.], [0., 1.], [1., 2.]])]
        qs = [np.array([.1, .2, .7])]
        args = (xs, qs, np.array([1.]), .01)
        error = check_grad(lambda t: objective(t, *args)[0], lambda t: objective(t, *args)[1], np.array([.2, -.3]))
        self.assertLess(error, 1e-6)

    def test_temperature_preserves_rank_and_changes_entropy(self):
        p = np.array([.7, .2, .1])
        scaled = temperature_scale([p], 2)[0]
        self.assertEqual(np.argmax(p), np.argmax(scaled))
        self.assertGreater(entropy(scaled), entropy(p))
        self.assertAlmostEqual(scaled.sum(), 1)

    def test_packed_objective_matches_weighted_variable_region_reference(self):
        rng = np.random.default_rng(5)
        xs = [rng.normal(size=(n,4)) for n in (2,5,3)]
        qs = [rng.dirichlet(np.ones(n)) for n in (2,5,3)]
        weights = np.array([.7,0.,.3]); theta = rng.normal(size=4)
        row_weights = np.repeat(weights,[2,5,3])
        args = (np.vstack(xs),np.array([0,2,7]),row_weights,np.concatenate(qs)*row_weights,weights,.03)
        expected = objective(theta,xs,qs,weights,.03)
        actual = packed_objective(theta,*args)
        self.assertAlmostEqual(expected[0],actual[0],places=12)
        np.testing.assert_allclose(expected[1],actual[1],atol=1e-12)
        error = check_grad(lambda t:packed_objective(t,*args)[0],lambda t:packed_objective(t,*args)[1],theta)
        self.assertLess(error,1e-6)

    def test_bootstrap_degenerate_data(self):
        np.testing.assert_allclose(bootstrap_mean_ci(np.ones(10)), [1, 1])


if __name__ == '__main__':
    unittest.main()
