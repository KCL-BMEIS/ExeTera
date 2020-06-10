from collections import defaultdict

import dataset
import utils

pfilename = '/home/ben/covid/patients_export_geocodes_20200604030001.csv'
tfilename = '/home/ben/covid/covid_test_export_20200604030001.csv'
asfilename = '/home/ben/covid/assessments_export_20200604030001.csv'
with open(pfilename) as f:
    p_ds = dataset.Dataset(f, stop_after=1, show_progress_every=1000000)
print(p_ds.names_)

p_keys = ('id', 'is_pregnant', 'ethnicity', 'is_smoker', 'smoker_status', 'smoked_years_ago')
with open(pfilename) as f:
    p_ds = dataset.Dataset(f, keys=p_keys, show_progress_every=1000000)

print(utils.build_histogram(p_ds.field_by_name('is_smoker')))
print(utils.build_histogram(p_ds.field_by_name('smoker_status')))
print(utils.build_histogram(p_ds.field_by_name('smoked_years_ago')))

smoking_pregnant_patients = set()
pregnant_patients = set()
p_ids = p_ds.field_by_name('id')
p_ips = p_ds.field_by_name('is_pregnant')
p_ismk = p_ds.field_by_name('is_smoker')
p_smks = p_ds.field_by_name('smoker_status')
for i_r in range(p_ds.row_count()):
    if p_ips[i_r] == 'True':
        pregnant_patients.add(p_ids[i_r])
        if p_ismk[i_r] == 'True' or p_smks[i_r] == 'yes':
            smoking_pregnant_patients.add(p_ids[i_r])

print('number of pregnant users:', len(pregnant_patients))
print('number of smoking pregnant users:', len(smoking_pregnant_patients))

# check against new tests
pregnant_new_tested_no = set()
pregnant_new_tested_yes = set()
with open(tfilename) as f:
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
        self.visited_hospital = False

    def set_result(self, result, location):
        if self.positive is None:
            self.positive = result
        else:
            self.positive = True if self.positive == True else result
        if self.visited_hospital is None:
            self.visited_hospital = location in ('hospital', 'back_from_hospital')
        else:
            self.visited_hospital =\
                self.visited_hospital or location in ('hospital', 'back_from_hospital')

with open(asfilename) as f:
    a_ds = dataset.Dataset(f, stop_after=1, show_progress_every=1000000)
print(a_ds.names_)

a_keys = ('patient_id', 'health_status', 'location', 'fever', 'persistent_cough', 'fatigue',
          'shortness_of_breath', 'diarrhoea', 'diarrhoea_frequency', 'delirium',
          'skipped_meals', 'abdominal_pain', 'chest_pain', 'hoarse_voice',
          'loss_of_smell', 'headache', 'headache_frequency', 'other_symptoms',
          'chills_or_shivers', 'eye_soreness', 'nausea', 'dizzy_light_headed',
          'red_welts_on_face_or_lips', 'blisters_on_feet', 'sore_throat',
          'unusual_muscle_pains', 'tested_covid_positive',)
with open(asfilename) as f:
    a_ds = dataset.Dataset(f, keys=a_keys, show_progress_every=1000000)
print(a_ds.names_)
tested_patients = defaultdict(TestResult)
a_pids = a_ds.field_by_name('patient_id')
a_tcps = a_ds.field_by_name('tested_covid_positive')
a_ltns = a_ds.field_by_name('location')
for i_r in range(a_ds.row_count()):
    if a_tcps[i_r] in ('no', 'yes'):
        tested_patients[a_pids[i_r]].set_result(a_tcps[i_r] == 'yes', a_ltns[i_r])

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
