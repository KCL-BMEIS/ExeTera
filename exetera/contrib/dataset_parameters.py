import sys

import h5py

from exetera.core import persistence as per

def h5dataset_summary(dataset):
    print(dataset.keys())
    for k in dataset.keys():
        print(f"{k}: {dataset[k].keys()}")
        print(k, len(dataset[k].keys()), len(per.get_reader(dataset[k]['id'])))


if __name__ == '__main__':
    with h5py.File(sys.argv[1], 'r') as ds:
        h5dataset_summary(ds)
