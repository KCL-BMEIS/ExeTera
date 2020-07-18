from hystore.core import dataset

pfilename = '/home/ben/covid/supplements_patients.csv'
afilename = '/home/ben/covid/supplements_assessments.csv'


with open(pfilename) as f:
    pds = dataset.Dataset(f, keys=('id', 'asmt_id'), show_progress_every=250000)

with open (afilename) as f:
    ads = dataset.Dataset(f, keys=('id', 'patient_id'), show_progress_every=250000)


ppids = pds.field_by_name('id')
paids = pds.field_by_name('asmt_id')

apids = ads.field_by_name('patient_id')
aaids = ads.field_by_name('id')


print("pids equal:", ppids == apids)
print("aids equal:", paids == aaids)