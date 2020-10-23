import numpy as np
import h5py


from exetera.core.session import Session
from exetera.core import persistence as per

source = '/home/ben/covid/ds_20200720_full.hdf5'
sink = '/home/ben/covid/ds_temp.hdf5'
print('ping')
with h5py.File(source, 'r') as hsrc:
    with h5py.File(sink, 'w') as hsnk:
        s = Session()
        asmts = hsrc['assessments']
        r_hcts = s.get_reader(asmts['had_covid_test'])
        r_pids = s.get_reader(asmts['patient_id'])

        flat_hcts = np.where(r_hcts[:] > 1, 1, 0)
        print(np.count_nonzero(flat_hcts == True), np.count_nonzero(flat_hcts == False))

        tests = hsrc['tests']
        tres = s.get_reader(tests['result'])
        filtpos = tres == b'positive'
        print(len(tres))

