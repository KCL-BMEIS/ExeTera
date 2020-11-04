import unittest

from io import BytesIO

import numpy as np
import h5py

from exetera.core.session import Session
from exetera.processing.test_type_from_mechanism import test_type_from_mechanism_v1


class TestTestTypeFromMechansim(unittest.TestCase):

    def test_test_type_from_mechanism_v1_numpy(self):
        s = Session()
        t_mech = np.asarray([-1, 0, 1, 2, 3,  4, -1, -1, 5, 6, 7, -1])
        t_mech_freetext =\
            np.asarray(["bloodxyz", "", "", "", "", "", "swabxyz", "selfxyz", "", "", "", "fingerxyz"])
        pcr1 = np.zeros(len(t_mech), dtype=np.bool)
        pcr2 = np.zeros(len(t_mech), dtype=np.bool)
        pcr3 = np.zeros(len(t_mech), dtype=np.bool)
        atb1 = np.zeros(len(t_mech), dtype=np.bool)
        atb2 = np.zeros(len(t_mech), dtype=np.bool)
        atb3 = np.zeros(len(t_mech), dtype=np.bool)
        test_type_from_mechanism_v1(s, t_mech, t_mech_freetext, pcr1, pcr2, pcr3, atb1, atb2, atb3)
        self.assertTrue(np.array_equal(pcr1, np.asarray([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.bool)))
        self.assertTrue(np.array_equal(pcr2, np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.bool)))
        self.assertTrue(np.array_equal(pcr3, np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool)))
        self.assertTrue(np.array_equal(atb1, np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], dtype=np.bool)))
        self.assertTrue(np.array_equal(atb2, np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)))
        self.assertTrue(np.array_equal(atb3, np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.bool)))

    def test_test_type_from_mechanism_v1_fields(self):
        bio = BytesIO()
        with h5py.File(bio, 'w') as hf:
            s = Session()
            t_mech = np.asarray([-1, 0, 1, 2, 3,  4, -1, -1, 5, 6, 7, -1])
            t_mech_f = s.create_numeric(hf, "t_mech", 'int8')
            t_mech_f.data.write(t_mech)

            t_mech_freetext =\
                np.asarray(["bloodxyz", "", "", "", "", "", "swabxyz", "selfxyz", "", "", "", "fingerxyz"])
            t_mech_freetext_f = s.create_indexed_string(hf, "t_mech_freetext")
            t_mech_freetext_f.data.write(t_mech_freetext)
            pcr1 = s.create_numeric(hf, 'pcr1', 'bool')
            pcr2 = s.create_numeric(hf, 'pcr2', 'bool')
            pcr3 = s.create_numeric(hf, 'pcr3', 'bool')
            atb1 = s.create_numeric(hf, 'atb1', 'bool')
            atb2 = s.create_numeric(hf, 'atb2', 'bool')
            atb3 = s.create_numeric(hf, 'atb3', 'bool')
            test_type_from_mechanism_v1(s, t_mech_f, t_mech_freetext_f, pcr1, pcr2, pcr3, atb1, atb2, atb3)
            self.assertTrue(
                np.array_equal(pcr1.data[:], np.asarray([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.bool)))
            self.assertTrue(
                np.array_equal(pcr2.data[:], np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.bool)))
            self.assertTrue(
                np.array_equal(pcr3.data[:], np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool)))
            self.assertTrue(
                np.array_equal(atb1.data[:], np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], dtype=np.bool)))
            self.assertTrue(
                np.array_equal(atb2.data[:], np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool)))
            self.assertTrue(
                np.array_equal(atb3.data[:], np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.bool)))
