from collections import defaultdict

import dataset
import utils


with open('/home/ben/covid/patients_export_geocodes_20200514030002.csv') as f:
    p_ds = dataset.Dataset(f, keys=('id', 'is_pregnant'), show_progress_every=1000000)

pregnant_patients = set()
p_ids = p_ds.field_by_name('id')
p_ips = p_ds.field_by_name('is_pregnant')
for i_r in range(p_ds.row_count()):
    if p_ips[i_r] == 'True':
        pregnant_patients.add(p_ids[i_r])

print(len(pregnant_patients))


# check against new tests
pregnant_new_tested_no = set()
pregnant_new_tested_yes = set()
with open('/home/ben/covid/covid_test_export_20200514030002.csv') as f:
    t_ds = dataset.Dataset(f, keys=('patient_id', 'result'))

t_pids = t_ds.field_by_name('patient_id')
t_rsts = t_ds.field_by_name('result')
for i_r in range(t_ds.row_count()):
    if t_pids[i_r] in pregnant_patients:
        if t_rsts[i_r] == 'positive':
            pregnant_new_tested_yes.add(t_pids[i_r])
        elif t_rsts[i_r] == 'negative':
            pregnant_new_tested_no.add(t_pids[i_r])

# check against old tests

class TestResult:
    def __init__(self):
        self.positive = None

    def set_result(self, result):
        if self.positive is None:
            self.positive = result
        else:
            self.positive = True if self.positive == True else result

with open('/home/ben/covid/assessments_export_20200514030002.csv') as f:
    a_ds = dataset.Dataset(f, keys=('patient_id', 'tested_covid_positive'), show_progress_every=1000000)

tested_patients = defaultdict(TestResult)
a_pids = a_ds.field_by_name('patient_id')
a_tcps = a_ds.field_by_name('tested_covid_positive')
for i_r in range(a_ds.row_count()):
    if a_tcps[i_r] in ('no', 'yes'):
        tested_patients[a_pids[i_r]].set_result(a_tcps[i_r] == 'yes')

print(len(tested_patients))

pregnant_old_tested_yes = set()
pregnant_old_tested_no = set()
for tp in tested_patients.items():
    if tp[0] in pregnant_patients:
        if tp[1].positive == True:
            pregnant_old_tested_yes.add(tp[0])
        else:
            pregnant_old_tested_no.add(tp[0])

print(f'new_tests: {len(pregnant_new_tested_yes)} positive, {len(pregnant_new_tested_no)} negative')
print(f'old tests: {len(pregnant_old_tested_yes)} positive, {len(pregnant_old_tested_no)} negative')

print(f'overall: {len(pregnant_new_tested_yes.union(pregnant_old_tested_yes))} positive')
print(f'overall: {len(pregnant_new_tested_no.union(pregnant_old_tested_no))} negative')
