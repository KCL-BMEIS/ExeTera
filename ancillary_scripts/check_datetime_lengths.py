import dataset
import utils

afilename = '/home/ben/covid/assessments_export_20200617030002.csv'


with open(afilename) as f:
    ds = dataset.Dataset(f, keys=('created_at', 'updated_at', 'date_test_occurred'),
                         show_progress_every=5000000)

cats = ds.field_by_name('created_at')
uats = ds.field_by_name('updated_at')
dtos = ds.field_by_name('date_test_occurred')

scats = set()
suats = set()
sdtos = set()

for i_r in range(ds.row_count()):
    scats.add(len(cats[i_r]))
    suats.add(len(uats[i_r]))
    sdtos.add(len(dtos[i_r]))
print(scats)
print(suats)
print(sdtos)