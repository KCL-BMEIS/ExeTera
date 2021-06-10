import unittest
from io import BytesIO

import numpy as np

from exetera.core.session import Session
from exetera.core.viewer import Viewer, ViewerMask


class TestUtils(unittest.TestCase):
    def test_viewer(self):
        bio = BytesIO()
        with Session() as s:
            src = s.open_dataset(bio, 'r+', 'src')
            df = src.create_dataframe('df')
            num = df.create_numeric('num', 'uint32')
            num.data.write([1, 2, 3, 4, 5, 6, 7])

            view = Viewer(df)
            mask = ViewerMask(np.where(num.data[:] > 3), np.array(['num']))
            view.mask = mask
            view['num'] == 7
            print(view['num'])
            print(view[:])


    def test_mask(self):
        idxlist = np.array([1, 3, 5, 7])
        clmlist = np.array(['a', 'b', 'c', 'd'])
        mask = ViewerMask(idxlist, clmlist)

        idx2 = np.array([1, 3, 6, 8])
        clm2 = np.array(['c', 'd', 'e', 'f'])
        msk2 = ViewerMask(idx2, clm2)

        m1 = mask & msk2
        self.assertEqual(m1.index.tolist(), [1, 3])
        self.assertEqual(m1.column.tolist(), ['c', 'd'])

        m2 = mask | msk2
        self.assertEqual(m2.index.tolist(), [1, 3, 5, 6, 7, 8])
        self.assertEqual(m2.column.tolist(), ['a', 'b', 'c', 'd', 'e', 'f'])

        m2 = m2 & ViewerMask(m2.index, np.array(['a', 'b']))
        self.assertEqual(m2.index.tolist(), [1, 3, 5, 6, 7, 8])
        self.assertEqual(m2.column.tolist(), ['a', 'b'])
