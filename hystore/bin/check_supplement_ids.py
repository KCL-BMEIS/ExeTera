from collections import defaultdict

from hytable.core import dataset

ds1 = None
with open('/home/ben/covid/patients_export_geocodes_20200702030001.csv') as f1:
    ds1 = dataset.Dataset(f1, keys=('id',))
pids1 = ds1.field_by_name('id')

ds2 = None
with open('/home/ben/OneDrive/supplement_paper/suppl_assessments_clean'
          '_w_hosp_treat_pred_covid_RC_17062020.csv') as f2:
    ds2 = dataset.Dataset(f2, keys=('id',))
pids2 = ds2.field_by_name('id')

spids1 = set(pids1)

spids2 = set(pids2)

print(len(spids2.difference(spids1)))
print(len(spids1.difference(spids2)))

print(len(spids1), len(set(spids1)))
print(len(spids2), len(set(spids2)))
counts = defaultdict(int)
for id in spids2:
    counts[id] += 1

hgram = defaultdict(int)
for v in counts.values():
    hgram[v] += 1

print(sorted(list(hgram.items())))
