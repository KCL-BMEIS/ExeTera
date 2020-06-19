import dataset
import utils

p_file_name = '/home/ben/covid/patients_export_geocodes_20200617030002.csv'


with open(p_file_name) as f:
    p_ds = dataset.Dataset(f, keys=('country_code',), show_progress_every=100000)

ccs = p_ds.field_by_name('country_code')
print(utils.build_histogram(ccs))
