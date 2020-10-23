import sys
import os

from collections import defaultdict

import numpy as np
from numba import jit, njit

from exetera.core import dataset


@njit
def _strequal(a, b, results):
    for i in range(len(a)):
        results[i] = a[i] == b[i]
    return results


def strequal(a, b):
    if a.shape != b.shape:
        raise ValueError(f"'a' and 'b' must be the same shape but are {a.shape} and {b.shape}")
    results = np.zeros(a.shape, dtype=np.bool)
    return _strequal(a, b, results)




if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: check_for_duplicates.py pattern <directory>")
        exit(1)

    filenames = sorted(fn for fn in os.listdir(sys.argv[1]) if sys.argv[2] in fn)
    for fn in filenames:
        print(fn)
        with open(os.path.join(sys.argv[1], fn)) as f:
            ds = dataset.Dataset(f, keys=('id',))
            ids = ds.field_by_name('id')
            ids = [b.encode() for b in ids]
            ids = np.asarray(ids, dtype='S32')
            cids = ids[:-1]
            nids = ids[1:]
            filter = strequal(cids, nids)
            print("  contiguous duplicates True/False:",
                  np.count_nonzero(filter == True), np.count_nonzero(filter == False))
            idcounts = defaultdict(int)
            for i in ids:
                idcounts[i] += 1
            uidcounts = defaultdict(int)
            for k, v in idcounts.items():
                uidcounts[v] += 1

            print("  overall duplicates:", sorted(list(uidcounts.items())))
            print()
            del ds
            del cids
            del nids
            del ids
            del filter
