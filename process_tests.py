from collections import defaultdict
import datetime

import dataset
import utils

from analytics import TestIndices, group_new_test_indices_by_patient

strformat = '%Y-%m-%d'
# t_file_name = '/home/ben/covid/covid_test_export_20200512030002.csv'
# a_file_name = '/home/ben/covid/assessments_export_20200512030002.csv'
# today = datetime.datetime.strptime("2020-05-12", strformat)
# t_file_name = '/home/ben/covid/covid_test_export_20200513030002.csv'
# a_file_name = '/home/ben/covid/assessments_export_20200513030002.csv'
# today = datetime.datetime.strptime("2020-05-13", strformat)
t_prev_file_name = '/home/ben/covid/covid_test_export_20200513030002.csv'
a_prev_file_name = '/home/ben/covid/assessments_export_20200513030002.csv'
prev_day = datetime.datetime.strptime("2020-05-13", strformat)
t_cur_file_name = '/home/ben/covid/covid_test_export_20200514030002.csv'
a_cur_file_name = '/home/ben/covid/assessments_export_20200514030002.csv'
cur_day = datetime.datetime.strptime("2020-05-14", strformat)


class LastActivity:
    def __init__(self):
        self.active = None

    def add(self, timestamp):
        if self.active is None:
            self.active = timestamp
        else:
            self.active = max(self.active, timestamp)

    def __repr__(self):
        return f"{self.active}"


# get stats and print delta for new tests
# ---------------------------------------

with open(t_prev_file_name) as f:
    t_prev_ds = dataset.Dataset(f)
t_prev_patients = group_new_test_indices_by_patient(t_prev_ds)

with open(t_cur_file_name) as f:
    t_cur_ds = dataset.Dataset(f)
t_cur_patients = group_new_test_indices_by_patient(t_cur_ds)

print(f'patients with new format tests ({prev_day}):', len(t_prev_patients))

print(f'patients with new format tests ({cur_day}):', len(t_cur_patients))

print(f'new format test delta:', len(t_cur_patients) - len(t_prev_patients))


def get_patients_with_old_format_tests(a_ds):
    apids = a_ds.field_by_name('patient_id')
    ahcts = a_ds.field_by_name('had_covid_test')
    atcps = a_ds.field_by_name('tested_covid_positive')
    auats = a_ds.field_by_name('updated_at')
    print('row count:', a_ds.row_count())

    apatients = defaultdict(int)
    for i_r in range(a_ds.row_count()):
        if ahcts[i_r] == 'True' or atcps[i_r] in ('waiting', 'no', 'yes'):
            apatients[apids[i_r]] += 1

    apatient_test_count = 0
    for k, v in apatients.items():
        if v > 0:
            apatient_test_count += 1

    return apatients


# get stats and print delta for old tests
# ---------------------------------------

a_keys = ('patient_id', 'created_at', 'updated_at', 'had_covid_test', 'tested_covid_positive')
with open(a_prev_file_name) as f:
    a_ds = dataset.Dataset(f, keys=a_keys, show_progress_every=5000000, stop_after=4999999)
a_prev_patients = get_patients_with_old_format_tests(a_ds)
del a_ds

with open(a_cur_file_name) as f:
    a_ds = dataset.Dataset(f, keys=a_keys, show_progress_every=5000000, stop_after=4999999)
a_cur_patients = get_patients_with_old_format_tests(a_ds)
del a_ds

print(f'patients with old format tests ({prev_day}):', len(a_prev_patients))

print(f'patients with old format tests ({cur_day}):', len(a_cur_patients))

print(f'new format test delta:', len(a_cur_patients) - len(a_prev_patients))


# build a dictionary of test counts by patient under the new system
# -----------------------------------------------------------------


s_prev_new_patients = set(t_prev_patients.keys())
s_cur_new_patients = set(t_cur_patients.keys())
s_prev_old_patients = set(a_prev_patients.keys())
s_cur_old_patients = set(a_cur_patients.keys())
s_prev_only_in_old = s_prev_old_patients.difference(s_prev_new_patients)
s_cur_only_in_old = s_cur_old_patients.difference(s_cur_new_patients)
print(len(s_prev_only_in_old))
print(len(s_cur_only_in_old))
print(len(s_cur_only_in_old) - len(s_prev_only_in_old))


def filter_duplicate_tests(ds, patients, threshold_for_diagnostic_print=1000000):
    tids = ds.field_by_name('id')
    pids = ds.field_by_name('patient_id')
    cats = ds.field_by_name('created_at')
    edates = ds.field_by_name('date_taken_specific')
    edate_los = ds.field_by_name('date_taken_between_start')
    edate_his = ds.field_by_name('date_taken_between_end')

    cleaned_patients = defaultdict(TestIndices)
    for p in patients.items():
        # print(p[0], len(p[1].indices))
        test_dates = set()
        for i_r in reversed(p[1].indices):
            test_dates.add((edates[i_r], edate_los[i_r], edate_his[i_r]))
        if len(test_dates) == 1:
            istart = p[1].indices[-1]
            # utils.print_diagnostic_row(f"{istart}", ds, istart, ds.names_)
            cleaned_patients[p[0]].add(istart)
        else:
            cleaned_entries = dict()
            for t in test_dates:
                cleaned_entries[t] = list()
            for i_r in reversed(p[1].indices):
                cleaned_entries[(edates[i_r], edate_los[i_r], edate_his[i_r])].append(i_r)

            for e in sorted(cleaned_entries.items(), key=lambda x: x[0]):
                last_index = e[1][0]
                if len(test_dates) > threshold_for_diagnostic_print:
                    utils.print_diagnostic_row(f"{p[0]}-{last_index}:", ds, last_index, ds.names_)
                cleaned_patients[p[0]].add(last_index)

    return cleaned_patients


t_cur_cleaned_patients = filter_duplicate_tests(t_cur_ds, t_cur_patients, threshold_for_diagnostic_print=10)