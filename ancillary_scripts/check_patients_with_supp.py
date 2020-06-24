
import dataset
import utils

pfilename = '/home/ben/covid/patients_export_geocodes_20200617030002.csv'

with open(pfilename) as f:
    ds = dataset.Dataset(f, keys=('vs_vitamin_d',), show_progress_every=1000000)

vits = ds.field_by_name('vs_vitamin_d')
print(utils.build_histogram(vits))
