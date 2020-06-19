
import dataset
import utils

pfilename = '/home/ben/covid/patients_export_geocodes_20200611030002.csv'
filename = '/home/ben/covid/covid_test_export_20200611030002.csv'
keys = ('country_code',)

with open(pfilename) as f:
    ds = dataset.Dataset(f, keys=keys, show_progress_every=100000)

    ccs = ds.field_by_name('country_code')
    print(utils.build_histogram(ccs))

with open(filename) as f:
    ds = dataset.Dataset(f, keys=keys, show_progress_every=100000)

    ccs = ds.field_by_name('country_code')
    print(utils.build_histogram(ccs))
