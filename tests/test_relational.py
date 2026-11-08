"""Neural contracts are optional when the base CPU-only dependencies are used."""
import importlib.util
import unittest
import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from attention.relational import (RegionNetwork, image_cross_entropy, fit_normalizer,
                                      observer_disagreement, rank_scores)


@unittest.skipUnless(HAS_TORCH, 'Install requirements-vision.txt for neural contracts')
class RelationalContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_permutation_equivariance_and_padding_isolation(self):
        for relation in ('self', 'uniform', 'learned'):
            model = RegionNetwork(5, 8, relation).eval()
            x = torch.randn(1, 4, 5)
            valid = torch.ones((1, 4), dtype=torch.bool)
            p = model(x, valid).softmax(-1)
            order = torch.tensor([2, 0, 3, 1])
            torch.testing.assert_close(model(x[:, order], valid).softmax(-1), p[:, order])
            padded = torch.cat([x, torch.full((1, 3, 5), 100.)], dim=1)
            mask = torch.tensor([[True]*4+[False]*3])
            actual = model(padded, mask).softmax(-1)
            torch.testing.assert_close(actual[:, :4], p)
            self.assertEqual(float(actual[:, 4:].sum().detach()), 0.)

    def test_batch_independence_and_single_background(self):
        model = RegionNetwork(5, 8, 'learned')
        x = torch.randn(2, 3, 5)
        valid = torch.tensor([[True, False, False], [True, True, True]])
        logits = model(x, valid)
        self.assertEqual(float(logits.softmax(-1)[0, 0].detach()), 1.)
        torch.testing.assert_close(logits[1], model(x[1:], valid[1:])[0])
        q = torch.tensor([[1., 0., 0.], [.2, .3, .5]])
        loss = image_cross_entropy(logits, q, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_equal_image_loss_not_pooled_events(self):
        logits = torch.tensor([[0., 0.], [0., 2.]])
        q = torch.tensor([[1., 0.], [0., 1.]])
        actual = image_cross_entropy(logits, q, torch.ones_like(q, dtype=torch.bool))
        expected = (np.log(2)+np.log1p(np.exp(-2)))/2
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_training_only_equal_image_normalizer(self):
        mean, std = fit_normalizer([np.array([[0., 7.]]), np.array([[2., 7.], [2., 7.]])])
        np.testing.assert_allclose(mean, [1., 7.])
        np.testing.assert_allclose(std, [1., 1.])

    def test_observer_js_and_rank_ties(self):
        self.assertAlmostEqual(observer_disagreement([[1, 0], [0, 1]]), 1.)
        self.assertAlmostEqual(observer_disagreement([[1, 1], [2, 2], [0, 0]]), 0.)
        self.assertIsNone(rank_scores([1, 0], [.5, .5])['spearman'])
        self.assertAlmostEqual(rank_scores([.7, .2, .1], [.7, .2, .1])['ndcg3'], 1.)
        tied = rank_scores([1, 0], [.5, .5])['ndcg3']
        self.assertAlmostEqual(tied, (1+1/np.log2(3))/2)

    def test_self_head_has_no_cross_region_logit_dependency(self):
        model = RegionNetwork(5, 8, 'self')
        x = torch.randn(1, 3, 5)
        mask = torch.ones((1, 3), dtype=torch.bool)
        original = model(x, mask)
        changed = x.clone(); changed[:, 1:] += 10
        torch.testing.assert_close(original[:, 0], model(changed, mask)[:, 0])


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec('torchvision') is not None,
                     'Torchvision required for mask pooling')
class VisualContracts(unittest.TestCase):
    def test_tiny_region_and_constant_features(self):
        from attention.visual import pool_regions
        labels = np.zeros((20, 20), int); labels[3, 7] = 1
        fmap = torch.full((4, 2, 2), 3.)
        result = pool_regions(fmap, labels, np.array([0, 1]))
        torch.testing.assert_close(result, torch.full((2, 4), 3.))


if __name__ == '__main__':
    unittest.main()
