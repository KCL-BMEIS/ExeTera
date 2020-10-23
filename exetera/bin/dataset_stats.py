import numpy as np
import h5py

from exetera.core.session import Session
from exetera.core.utils import Timer


with h5py.File('/home/ben/covid/ds_20200914_full.hdf5') as hf:
    s = Session()
    with Timer('a_pids'):
        a_pids = s.get(hf['assessments']['patient_id']).data[:50000000]

    with Timer('save'):
        np.savez('/home/ben/covid/big_a_pids.npz', values=a_pids)

    with Timer('reload'):
        a_pids2 = np.load('/home/ben/covid/big_a_pids.npz')
        v = a_pids2['values']

    with Timer('save'):
        np.save('/home/ben/covid/big_a_pids.npy', a_pids)
        del a_pids

    with Timer('reload'):
        a_pids2 = np.load('/home/ben/covid/big_a_pids.npy')
