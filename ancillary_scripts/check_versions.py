import dataset
import pipeline

patients_filename = '/home/ben/covid/patients_export_geocodes_20200423050002.csv'
assessments_filename = '/home/ben/covid/assessments_export_20200423050002.csv'
#fn = '/home/ben/covid/assessments_short.csv'
print(f'loading {patients_filename}')
with open(patients_filename) as f:
    ds = dataset.Dataset(f, progress=True)

print(pipeline.build_histogram_from_list(ds.field_by_name('version')))

print(f'loading {assessments_filename}')
with open(assessments_filename) as f:
    ds = dataset.Dataset(f, progress=True)

print(pipeline.build_histogram_from_list(ds.field_by_name('version')))