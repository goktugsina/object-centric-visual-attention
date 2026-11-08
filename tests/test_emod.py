import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from attention.emod import load_emod


class EmodCoordinates(unittest.TestCase):
    def test_source_axes_duration_and_missing_images(self):
        # Non-square image and asymmetric coordinates catch an axis reversal.
        subject = dict(fix_x=np.array([2., 3., 0., 4.]),
                       fix_y=np.array([7., 8., 2., 5.]),
                       fix_duration=np.array([100., 99., 200., np.nan]))
        records = [dict(img='available.jpg', subjects=[subject]),
                   dict(img='missing.jpg', subjects=[subject])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'stimuli').mkdir()
            Image.new('RGB', (10, 8)).save(root/'stimuli/available.jpg')
            with patch('attention.emod.loadmat', return_value={'fixations1019': records}):
                data, audit = load_emod(root)
        x, y = data['available.jpg']['subjects'][0]
        np.testing.assert_array_equal(x, [6]); np.testing.assert_array_equal(y, [1])
        self.assertEqual(audit['retained_fixations'], 1)
        self.assertEqual(audit['rejected_duration'], 2)
        self.assertEqual(audit['rejected_coordinates'], 1)
        self.assertEqual(audit['missing_images'], ['missing.jpg'])


if __name__ == '__main__': unittest.main()
