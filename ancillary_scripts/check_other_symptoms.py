import dataset

with open('/home/ben/covid/assessments_export_20200428050002.csv') as f:
    ds = dataset.Dataset(f, keys=('patient_id', 'updated_at', 'other_symptoms'),
                         progress=True)
                         # progress=True, stop_after=999999)

other = ds.field_by_name('other_symptoms')

empty = 0
for i_r in range(len(other)):
    if other[i_r] == '':
        empty += 1

print('empty rows:', empty)
