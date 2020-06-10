from collections import defaultdict

import dataset
import utils
import analytics

filename = '/home/ben/covid/assessments_20200413050002_clean.csv'

a_keys = ('id', 'patient_id', 'created_at', 'updated_at',
          'had_covid_test', 'had_covid_test_clean',
          'tested_covid_positive', 'tested_covid_positive_clean')
with open(filename) as f:
    a_ds = dataset.Dataset(f, keys=('id', 'patient_id', 'created_at', 'updated_at',
                             'had_covid_test', 'had_covid_test_clean',
                             'tested_covid_positive', 'tested_covid_positive_clean'),
                    show_progress_every=5000000)

print('sorting')
a_ds.sort(('created_at',))
a_pids = a_ds.field_by_name('patient_id')
a_cats = a_ds.field_by_name('created_at')
a_hcts = a_ds.field_by_name('had_covid_test')
a_hctcs = a_ds.field_by_name('had_covid_test_clean')
a_tcps = a_ds.field_by_name('tested_covid_positive')
a_tcpcs = a_ds.field_by_name('tested_covid_positive_clean')


print('building per-patient indices')
a_patients = defaultdict(analytics.TestIndices)
for i_r in range(a_ds.row_count()):
    a_patients[a_pids[i_r]].add(i_r)


results = {
    '': {'': 0, 'waiting': 0, 'no': 0, 'yes': 0},
    'False': {'': 0, 'waiting': 0, 'no': 0, 'yes': 0},
    'True': {'': 0, 'waiting': 0, 'no': 0, 'yes': 0},
}

print('checking assessments')
for kv, kp in a_patients.items():

    anomolous = False
    for i_r in kp.indices:
        if a_hctcs[i_r] != 'True':
            if a_tcpcs[i_r] != '':
                anomolous = True
        else:
            if a_tcpcs[i_r] == '':
                anomolous = True
        results[a_hctcs[i_r]][a_tcpcs[i_r]] += 1
    if anomolous:
        print(kv)
        for i_r in kp.indices:
            utils.print_diagnostic_row(f"{i_r}", a_ds, i_r, a_keys)


for r in results.items():
    print(r[0], r[1])