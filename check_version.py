import dataset
import pipeline

fn = '/home/ben/covid/patients_export_geocodes_20200421050002.csv'

print('loading file')
with open(fn) as f:
    ds = dataset.Dataset(f)
ds.sort(('id',))
print('done')

print(pipeline.build_histogram(ds.fields_, ds.field_to_index('version')))
