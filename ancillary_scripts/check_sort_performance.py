
import time

import dataset

filename = '/home/ben/covid/assessments_export_20200514030002.csv'

keys = ('id', 'patient_id', 'created_at', 'updated_at', 'tested_covid_positive')
with open(filename) as f:
    ds = dataset.Dataset(f, keys=keys,
                         stop_after=5000000)
t0 = time.time()
ds.sort(('patient_id', 'created_at'))
print('sorted:', time.time() - t0)
lens = set()
pids = ds.field_by_name('patient_id')
cats = ds.field_by_name('created_at')
firsts = list()
for i_r in range(ds.row_count()):
    l = len(cats[i_r])
    if l not in lens:
        firsts.append(cats[i_r])
        lens.add(len(cats[i_r]))
print(lens)
print(firsts)
del ds

with open(filename) as f:
    ds = dataset.Dataset(f, keys=keys,
                         stop_after=5000000)
t0 = time.time()
ds.sort('created_at')
ds.sort('patient_id')
print('sorted:', time.time() - t0)
lens = set()
uats = ds.field_by_name('updated_at')
firsts = list()
for i_r in range(ds.row_count()):
    l = len(uats[i_r])
    if l not in lens:
        firsts.append(uats[i_r])
        lens.add(len(uats[i_r]))
print(lens)
print(firsts)
del ds

with open(filename) as f:
    ds = dataset.Dataset(f, keys=keys,
                        stop_after=25000000)
t0 = time.time()
ds.sort(('updated_at',))
print(time.time() - t0)

t0 = time.time()
ds._apply_permutation(ds.index_, ds.field_by_name('patient_id'))
print(time.time() - t0)

pids = ds.field_by_name('patient_id')
t0 = time.time()
count_empty = 0
for i_r in range(ds.row_count()):
    if pids[i_r] == '':
        count_empty += 1
print(time.time() - t0)

t0 = time.time()
count_empty = 0
for i_r in ds.index_:
    if pids[i_r] == '':
        count_empty += 1
print(time.time() - t0)

