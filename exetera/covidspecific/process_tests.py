# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import numpy as np

from exetera.core import dataset, utils

from exetera.processing.analytics import group_new_test_indices_by_patient, get_patients_with_old_format_tests, \
    filter_duplicate_new_tests
from exetera.processing.convert_old_assessments import ConvertOldAssessmentsV1

strformat = '%Y-%m-%d'
t_file_name = '/home/ben/covid/covid_test_export_20200601030001.csv'
a_file_name = '/home/ben/covid/assessments_export_20200601030001.csv'
cur_day = datetime.datetime.strptime("2020-05-18", strformat)


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


def sort_by_test_index_count(test_indices_by_patient):
    sorted_patient_text_index_pairs = sorted([t for t in test_indices_by_patient.items()],
                                             key=lambda t: len(t[1].indices), reverse=True)
    return sorted_patient_text_index_pairs


# start
with open(t_file_name) as f:
    t_ds = dataset.Dataset(f)
t_dtss = t_ds.field_by_name('date_taken_specific')
t_patients = group_new_test_indices_by_patient(t_ds)


# get stats and print delta for old tests
# ---------------------------------------

# a_keys = ('id', 'patient_id', 'country_code', 'created_at', 'updated_at', 'version', 'had_covid_test', 'tested_covid_positive')
a_keys = ('patient_id', 'updated_at', 'had_covid_test', 'tested_covid_positive')
with open(a_file_name) as f:
    a_ds = dataset.Dataset(f, keys=a_keys,
                           show_progress_every=5000000)
                           # show_progress_every=5000000, stop_after=1000000)
print('sorting')
a_ds.sort(keys='updated_at')

# a_ids = a_ds.field_by_name('id')
a_pids = a_ds.field_by_name('patient_id')
# a_cats = a_ds.field_by_name('created_at')
a_uats = a_ds.field_by_name('updated_at')
# a_vsns = a_ds.field_by_name('version')
# a_ccs = a_ds.field_by_name('country_code')
a_hcts = a_ds.field_by_name('had_covid_test')
a_tcps = a_ds.field_by_name('tested_covid_positive')
a_patients = get_patients_with_old_format_tests(a_ds)


print('patients with old tests:', len(a_patients))
print('patients with new tests:', len(t_patients))

# build a dictionary of test counts by patient under the new system
# -----------------------------------------------------------------


s_new_patients = set(t_patients.keys())
s_old_patients = set(a_patients.keys())
s_only_in_old = s_old_patients.difference(s_new_patients)
s_only_in_new = s_new_patients.difference(s_old_patients)
s_in_both = s_old_patients.intersection(s_new_patients)
print('only in old:', len(s_only_in_old))
print('only in new:', len(s_only_in_new))
print('in_both:', len(s_in_both))

t_cleaned_patients = filter_duplicate_new_tests(t_ds, t_patients, threshold_for_diagnostic_print=10)

t_cleaned_patient_entries = sort_by_test_index_count(t_cleaned_patients)

# p_0 = t_cleaned_patient_entries[0]

print(utils.build_histogram([len(x[1].indices) for x in t_cleaned_patient_entries]))


# a_new_rows = dict({'id': list(), 'patient_id': list(), 'created_at': list(), 'updated_at': list(),
#                    'version': list(), 'country_code': list(), 'result': list(), 'mechanism': list(),
#                    'date_taken_specific': list()})
#
# # create new test rows for patients that have had only old tests
# for pk, pv in a_patients.items():
#     if pk in s_only_in_old:
#         if pv.seen_negative + pv.seen_positive == 1:
#             if pv.seen_negative:
#                 result = 'negative'
#             else:
#                 result = 'positive'
#             # take the first entry
#             i_r = pv.indices[0]
#             a_new_rows['id'].append(a_ids[i_r])
#             a_new_rows['patient_id'].append(a_pids[i_r])
#             a_new_rows['created_at'].append(a_cats[i_r])
#             a_new_rows['updated_at'].append(a_uats[i_r])
#             a_new_rows['version'].append(a_vsns[i_r])
#             a_new_rows['country_code'].append(a_vsns[i_r])
#             a_new_rows['result'].append(result)
#             a_new_rows['date_taken_specific'].append(a_cats[i_r])

# print('adapted test count:', len(a_new_rows['id']))

value_map = {
    '': 0,
    'waiting': 1,
    'no': 2,
    'yes': 3
}
fn = ConvertOldAssessmentsV1(a_ds, t_ds, value_map)
results = fn(a_patients, s_new_patients, np.zeros(a_ds.row_count(), dtype=np.uint32))

# for i in range(100):
#     print(results['id'][i], results['patient_id'][i], results['created_at'][i],
#           results['updated_at'][i],
#           results['result'][i], results['date_taken_specific'][i])
#     for i_n in a_patients[results['patient_id'][i]].indices:
#         utils.print_diagnostic_row(f"{i_n}", a_ds, i_n, keys=a_keys)
#

dest_row_count = t_ds.row_count() + len(results['id'])
destination_tests = dict()
for n, f in zip(t_ds.names_, t_ds.fields_):
    if isinstance(f, list):
        field = [None] * dest_row_count
    else:
        field = np.zeros(dest_row_count, dtype=f.dtype)
    field[:t_ds.row_count()] = f
    field[t_ds.row_count():] = results[n]



print(len(results['id']))

exit()

# visually compare tests for patients that have both old and new tests
for p in t_cleaned_patient_entries:
    if len(p[1].indices) == 4:
        print(p[0])
        #get indices for the given patient sorted by test date
        t_sorted_indices = sorted(p[1].indices, key=lambda t: t_dtss[t])
        for s in t_sorted_indices:
            utils.print_diagnostic_row(f"{p[0]}-{s}", t_ds, s, t_ds.names_)

        # sort assessments belonging to that patient by date
        a_indices = list()
        for i_r in range(a_ds.row_count()):
            if a_pids[i_r] == p[0]:
                if a_hcts[i_r] == 'True' or a_tcps[i_r] in ('no', 'yes'):
                    a_indices.append(i_r)

        a_sorted_indices = sorted(a_indices, key=lambda t: a_uats[t])
        for s in a_sorted_indices:
            utils.print_diagnostic_row(f"{p[0]}-{s}", a_ds, s, a_keys)
