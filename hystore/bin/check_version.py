import sys

from hytable.core import dataset

def enumerate_versions(filenames):

    for fn in filenames:
        print(fn)
        with open(fn) as f:
            ds = dataset.Dataset(f, keys=('version',), show_progress_every=1000000)
        sversions = set(ds.field_by_name('version'))
        print(sorted(list(sversions)))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: check_version <one or more filenames>")
        exit(1)
    enumerate_versions(sys.argv[1:])
