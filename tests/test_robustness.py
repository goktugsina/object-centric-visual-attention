import unittest
import numpy as np
from attention.robustness import observer_target, project, permute_semantics, omit_semantics


class RobustnessContracts(unittest.TestCase):
    def scene(self):
        x=np.arange(54,dtype=float).reshape(3,18);x[0,6:]=0
        return {'name':'a.jpg','features':x,'active_ids':np.array([0,1,2])}

    def test_equal_observers_do_not_weight_long_sequences_more(self):
        np.testing.assert_allclose(observer_target([[90,10],[0,2],[0,0]]),[.45,.55])
        with self.assertRaises(ValueError):observer_target([[0,0]])

    def test_feature_projection_does_not_modify_source(self):
        scene=self.scene();before=scene['features'].copy()
        result=project([scene],[1,7,9]);result[0]['features'][0,0]=-1
        np.testing.assert_array_equal(scene['features'],before)

    def test_permutation_preserves_geometry_background_and_attribute_vectors(self):
        scene=self.scene();result=permute_semantics([scene],1)[0]['features']
        np.testing.assert_array_equal(result[:,:6],scene['features'][:,:6])
        np.testing.assert_array_equal(result[0],scene['features'][0])
        self.assertEqual(sorted(map(tuple,result[1:,6:])),sorted(map(tuple,scene['features'][1:,6:])))

    def test_omission_endpoints_and_no_mutation(self):
        scene=self.scene();before=scene['features'].copy()
        np.testing.assert_array_equal(omit_semantics([scene],0,1)[0]['features'],before)
        missing=omit_semantics([scene],1,1)[0]['features']
        np.testing.assert_array_equal(missing[:,6:],0)
        np.testing.assert_array_equal(missing[:,:6],before[:,:6])
        np.testing.assert_array_equal(scene['features'],before)


if __name__=='__main__':unittest.main()
