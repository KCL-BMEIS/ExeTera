from collections import defaultdict
import datetime
import time

import dataset
import utils
import analytics


strformat = '%Y-%m-%d'
# t_file_name = '/home/ben/covid/covid_test_export_20200512030002.csv'
# a_file_name = '/home/ben/covid/assessments_export_20200512030002.csv'
# today = datetime.datetime.strptime("2020-05-12", strformat)
# t_file_name = '/home/ben/covid/covid_test_export_20200513030002.csv'
# a_file_name = '/home/ben/covid/assessments_export_20200513030002.csv'
# today = datetime.datetime.strptime("2020-05-13", strformat)
t_prev_file_name = '/home/ben/covid/covid_test_export_20200601030001.csv'
a_prev_file_name = '/home/ben/covid/assessments_export_20200601030001.csv'
prev_day = datetime.datetime.strptime("2020-06-01", strformat)
t_cur_file_name = '/home/ben/covid/covid_test_export_20200604030001.csv'
a_cur_file_name = '/home/ben/covid/assessments_export_20200604030001.csv'
cur_day = datetime.datetime.strptime("2020-06-04", strformat)


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
t_prev_patients = analytics.group_new_test_indices_by_patient(t_prev_ds)

print(utils.build_histogram(t_prev_ds.field_by_name('result')))

with open(t_cur_file_name) as f:
    t_cur_ds = dataset.Dataset(f)
t_cur_patients = analytics.group_new_test_indices_by_patient(t_cur_ds)

print(f'patients with new format tests ({prev_day}):', len(t_prev_patients))

print(f'patients with new format tests ({cur_day}):', len(t_cur_patients))

print(f'new format test delta:', len(t_cur_patients) - len(t_prev_patients))

# get stats and print delta for old tests
# ---------------------------------------

a_keys = ('patient_id', 'created_at', 'updated_at', 'had_covid_test', 'tested_covid_positive')
with open(a_prev_file_name) as f:
    a_ds = dataset.Dataset(f, keys=a_keys, show_progress_every=5000000)
t0 = time.time()
a_ds.sort(('patient_id', 'updated_at'))
print('tuple sort:', time.time() - t0)
t0 = time.time()
a_prev_patients = analytics.get_patients_with_old_format_tests(a_ds)
del a_ds

with open(a_cur_file_name) as f:
    a_ds = dataset.Dataset(f, keys=a_keys, show_progress_every=5000000)
a_cur_patients = analytics.get_patients_with_old_format_tests(a_ds)
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

    cleaned_patients = defaultdict(analytics.TestIndices)
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


def link_new_tests_to_old_tests(t_ds, a_ds):
    pass


patients = defaultdict(analytics.TestIndices)
for i_r in range(tds.row_count()):
    patients[pids[i_r]].add(i_r)

# report patient count under new system
print('patients with tests - new system:', len(patients))

s_old_patients = set(apatients.keys())
s_new_patients = set(patients.keys())
only_in_old = s_old_patients.difference(s_new_patients)
print('only in old:', len(only_in_old))
print('only in new:', len(s_new_patients.difference(s_old_patients)))


# get time since last activity for people who registered tests in the old system but not the new one

old_patient_activity = defaultdict(LastActivity)
for i_r in range(a_ds.row_count()):
    if apids[i_r] in only_in_old:
        old_patient_activity[apids[i_r]].add(auats[i_r])

inactive_since = list()
for k, v in old_patient_activity.items():
    day = utils.timestamp_to_day(v.active)
    inactive_since.append((cur_day - datetime.datetime.strptime(day, strformat)).days)
sorted_inactivity = sorted(utils.build_histogram(inactive_since))
print("patient inactivity - with old tests but not with new:", sorted_inactivity)



patients = [p for p in patients.items()]
patients = sorted(patients, key=lambda x: len(x[1].indices), reverse=True)
patient_test_counts = [(p[0], len(p[1].indices)) for p in patients]
print(utils.build_histogram(patient_test_counts, tx=lambda x: x[1]))

cleaned_patients = defaultdict(analytics.TestIndices)
for p in patients:
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
            # utils.print_diagnostic_row(f"{last_index}", ds, last_index, ds.names_)
            cleaned_patients[p[0]].add(last_index)

cleaned_patients = [p for p in cleaned_patients.items()]
cleaned_patient_test_counts = sorted([(p[0], len(p[1].indices)) for p in cleaned_patients], key=lambda x: x[1], reverse=True)
print(utils.build_histogram(cleaned_patient_test_counts, tx=lambda x: x[1]))

# generate histogram of inexact test date interval durations
# ----------------------------------------------------------
date_deltas = defaultdict(int)
exact_count = 0
approx_count = 0
both_count = 0
for p in cleaned_patients:
    # just for diagnostic purposes
    # if len(p[1].indices) > 2:
    #     print(p[0], len(p[1].indices))
    #     for i_t in p[1].indices:
    #         utils.print_diagnostic_row(f"{i_t}", ds, i_t, ds.names_)

    for i_t in p[1].indices:
        if edates[i_t] != '' and edate_los[i_t] != '':
            both_count += 1
        else:
            if edates[i_t] != '':
                exact_count += 1
            if edate_los[i_t] != '':
                approx_count += 1
                date_deltas[tids[i_t]] = (datetime.datetime.strptime(edate_his[i_t], strformat) - datetime.datetime.strptime(edate_los[i_t], strformat)).days

date_hgram = utils.build_histogram(d for d in date_deltas.values())
date_hgram = sorted(date_hgram, key=lambda x: x[0])
print(date_hgram)


print("exact count:", exact_count)
print("approx count:", approx_count)
print("both count:", both_count)
print("total:", exact_count + approx_count + both_count)
# selected_test_per_patient = [(p[0], p[1].indices[0]) for p in patients]
# print(sum(len(p[1].indices) > 2 for p in patients))



