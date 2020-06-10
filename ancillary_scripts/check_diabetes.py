
import dataset
import utils

pfilename = '/home/ben/covid/patients_export_geocodes_20200601030001.csv'

with open(pfilename) as f:
    ds = dataset.Dataset(f, keys=('has_diabetes',), show_progress_every=1000000)

diabetes_count = 0
adbs = ds.field_by_name('has_diabetes')
for a in adbs:
    if a == 'True':
        diabetes_count += 1
print(diabetes_count)
print(utils.build_histogram(adbs))


