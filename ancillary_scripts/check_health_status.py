import dataset
import pipeline

# fn = '/home/ben/covid/assessments_export_20200421050002.csv'
fn = '/home/ben/covid/assessments_short.csv'
print('loading file')
with open(fn) as f:
    ds = dataset.Dataset(f, progress=True)
ds.sort(('patient_id', 'updated_at'))
print('done')

print(pipeline.build_histogram(ds.fields_, ds.field_to_index('health_status')))

for ir, r in enumerate(ds.fields_):
    if r[1] == '1fa25a81ffff33aedb467d71af2ecba9':
        print(r[0], r[1], r[3], r[6])
