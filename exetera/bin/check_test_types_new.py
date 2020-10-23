import numpy as np
import h5py

from exetera.core import persistence
from exetera.processing.test_type_from_mechanism import test_type_from_mechanism_v1

filename = '/home/ben/covid/ds_20200824_pt.hdf5'
filenametemp = '/home/ben/covid/ds_temp.hdf5'
ds = persistence.DataStore()
with h5py.File(filename, 'r') as src:
    with h5py.File(filenametemp, 'w') as tmp:
        tests = src['tests']
        print(tests.keys())
        mechanism = tests['mechanism']
        mechanism_free = tests['mechanism_freetext']
        test_result = ds.get_reader(tests['result'])[:]
        negative_test = test_result == 3
        positive_test = test_result == 4
        pcr_standard = ds.get_numeric_writer(tmp, 'pcr_standard', 'bool')
        pcr_strong_inferred = ds.get_numeric_writer(tmp, 'pcr_strong_inferred', 'bool')
        pcr_weak_inferred = ds.get_numeric_writer(tmp, 'pcr_weak_inferred', 'bool')
        anti_standard = ds.get_numeric_writer(tmp, 'antibody_standard', 'bool')
        anti_strong_inferred = ds.get_numeric_writer(tmp, 'antibody_strong_inferred', 'bool')
        anti_weak_inferred = ds.get_numeric_writer(tmp, 'antibody_weak_inferred', 'bool')
        test_type_from_mechanism_v1(ds, mechanism, mechanism_free,
                                    pcr_standard, pcr_strong_inferred, pcr_weak_inferred,
                                    anti_standard, anti_strong_inferred, anti_weak_inferred)
        print('tests:', len(ds.get_reader(tmp['pcr_standard'])))
        pcr_0 = ds.get_reader(tmp['pcr_standard'])[:]
        pcr_1 = ds.get_reader(tmp['pcr_strong_inferred'])[:]
        pcr_2 = ds.get_reader(tmp['pcr_weak_inferred'])[:]
        atb_0 = ds.get_reader(tmp['antibody_standard'])[:]
        atb_1 = ds.get_reader(tmp['antibody_strong_inferred'])[:]
        atb_2 = ds.get_reader(tmp['antibody_weak_inferred'])[:]
        for f in (('negative', negative_test), ('positive', positive_test)):
            print(f[0])
            print('  pcr_standard', np.count_nonzero(pcr_0 & f[1]))
            print('  pcr_strong_inferred', np.count_nonzero(pcr_1 & f[1]))
            print('  pcr_weak_inferred', np.count_nonzero(pcr_2 & f[1]))
            print('  antibody_standard', np.count_nonzero(atb_0 & f[1]))
            print('  antibody_strong_inferred', np.count_nonzero(atb_1 & f[1]))
            print('  antibody_weak_inferred', np.count_nonzero(atb_2 & f[1]))
            print('  both strong', np.count_nonzero(pcr_1 & atb_1 & f[1]))
            print('  both weak', np.count_nonzero(pcr_2 & atb_2 & f[1]))
        all_pcr = pcr_0 | pcr_1 | pcr_2
        all_atb = atb_0 | atb_1 | atb_2
        print('all antibody tests:', sum(all_atb))
        print('positive_antibody_tests:', sum(all_atb & positive_test))
        print('negative_antibody_tests:', sum(all_atb & negative_test))
        print('pcr p/n ratio',
              np.count_nonzero(all_pcr & positive_test) / np.count_nonzero(all_pcr & negative_test))
        print('atb p/n ratio',
              np.count_nonzero(all_atb & positive_test) / np.count_nonzero(all_atb & negative_test))

