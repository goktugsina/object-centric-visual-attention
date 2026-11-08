import unittest
import numpy as np
from attention.image_only import box_partition,region_features,pixel_information_gain,probability_map


class ImageOnlyContracts(unittest.TestCase):
    def test_confidence_and_stable_ties_resolve_overlaps(self):
        boxes=[[0,0,3,3],[1,1,4,4]]
        self.assertEqual(box_partition(boxes,[.8,.9],4,4)[1,1],2)
        self.assertEqual(box_partition(boxes,[.9,.9],4,4)[1,1],1)
        self.assertEqual(box_partition(boxes,[.8,.9],4,4)[0,3],0)

    def test_empty_detections_produce_valid_image_only_distribution(self):
        d={'width':4,'height':3,'boxes':[],'scores':[],'classes':[]}
        labels,active,area,x=region_features(d,91)
        np.testing.assert_array_equal(active,[0]);np.testing.assert_allclose(area,[1])
        self.assertEqual(x.shape,(1,98));self.assertEqual(x[0,5],1)
        np.testing.assert_allclose(probability_map(labels,active,[1]),np.full((3,4),1/12))

    def test_area_probabilities_equal_uniform_pixels_under_any_partition(self):
        for area in ([.1,.9],[.2,.3,.5]):
            q=np.zeros(len(area));q[0]=1
            self.assertAlmostEqual(pixel_information_gain(q,area,area),0)

    def test_pixel_density_normalizes_and_matches_score(self):
        labels=np.array([[0,1],[1,1]]);p=np.array([.4,.6]);area=np.array([.25,.75])
        density=probability_map(labels,[0,1],p)
        self.assertAlmostEqual(density.sum(),1)
        self.assertAlmostEqual(pixel_information_gain([.5,.5],p,area),.5*np.log2(density[0,0]*4)+.5*np.log2(density[1,1]*4))


if __name__=='__main__':unittest.main()
