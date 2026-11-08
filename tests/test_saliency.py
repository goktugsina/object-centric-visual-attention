import unittest
import numpy as np
from attention.saliency import normalize_density, fixation_target, pixel_scores, auc_judd
from attention.image_only import probability_map, pixel_information_gain


class PixelContracts(unittest.TestCase):
    def test_uniform_reference_and_constant_conventions(self):
        target = fixation_target([(np.array([0, 1, 1]), np.array([0, 1, 1]))], 4, 3)
        result = pixel_scores(np.ones((3, 4)), target)
        self.assertAlmostEqual(result['ig_uniform_bits'], 0.)
        self.assertAlmostEqual(result['auc_judd'], .5)
        self.assertEqual(result['nss'], 0.)
        self.assertEqual(result['cc'], 0.)

    def test_pixel_likelihood_matches_region_formula_with_repeated_events(self):
        labels = np.array([[0, 1], [1, 1]])
        p = np.array([.4, .6]); area = np.array([.25, .75])
        target = fixation_target([(np.array([0, 0, 1]), np.array([0, 0, 1]))], 2, 2)
        actual = pixel_scores(probability_map(labels, [0, 1], p), target)['ig_uniform_bits']
        self.assertAlmostEqual(actual, pixel_information_gain([2/3, 1/3], p, area))

    def test_perfect_reverse_auc_and_ties(self):
        mask = np.array([[True, True], [False, False]])
        self.assertAlmostEqual(auc_judd([[3, 4], [1, 2]], mask), 1.)
        # Judd uses positive thresholds: ROC (0,0),(1,.5),(1,1).
        # Its finite-grid trapezoid differs from all-threshold rank AUC (zero).
        self.assertAlmostEqual(auc_judd([[1, 2], [3, 4]], mask), .25)
        self.assertAlmostEqual(auc_judd(np.ones((2, 2)), mask), .5)

    def test_equal_observer_score_is_not_event_pooled(self):
        p = normalize_density([[.6, .2], [.1, .1]])
        subjects = [(np.array([0]*10), np.array([0]*10)), (np.array([1]), np.array([1]))]
        target = fixation_target(subjects, 2, 2)
        r = pixel_scores(p, target)
        self.assertAlmostEqual(r['equal_observer_ig_bits'], (np.log2(.6*4)+np.log2(.1*4))/2)
        self.assertNotAlmostEqual(r['ig_uniform_bits'], r['equal_observer_ig_bits'])

    def test_invalid_map_and_coordinates_rejected(self):
        for p in ([[-1, 2]], [[np.nan, 1]], [[0, 0]]):
            with self.assertRaises(ValueError): normalize_density(p)
        with self.assertRaises(ValueError):
            fixation_target([(np.array([2]), np.array([0]))], 2, 2)


if __name__ == '__main__': unittest.main()
