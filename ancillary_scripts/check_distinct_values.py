import numpy as np

import h5py

import persistence


with h5py.File('/home/ben/covid/ds_20200624_sorted.hdf5', 'r') as hf:

    for f in ('isolation_little_interaction', 'isolation_lots_of_people',
              'isolation_healthcare_provider'):
        distinct_values = np.unique(
            persistence.get_reader(hf['assessments'][f])[:],
            return_counts=True)
        print(f, distinct_values[0][:10], distinct_values[1][:10])
