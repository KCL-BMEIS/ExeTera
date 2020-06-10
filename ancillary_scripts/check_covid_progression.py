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

from collections import defaultdict

import numpy as np

import dataset
import data_schemas
import parsing_schemas
import pipeline
from processing.assessment_merge import MergeAssessmentRows, CalculateMergedFieldCount
from processing.inconsistent_testing import CheckTestingConsistency

#filename = '/home/ben/covid/assessments_short.csv'
import processing.covid_test
import utils

filename = '/home/ben/covid/assessments_export_20200508030002.csv'
#filename = '/home/ben/covid/assessments_20200413050002_clean_bak.csv'

def progress():
    i = 0
    if i > 0 and i % 1000 == 0:
        print('.', end='')
        if i % 10000 == 0:
            print('i')
    i += 1
    yield

print(f"loading {filename}")
with open(filename) as f:
    ds = dataset.Dataset(f, data_schemas.DataSchema(1).assessment_categorical_maps,
                         keys=['id', 'patient_id', 'created_at', 'updated_at', 'had_covid_test', 'tested_covid_positive'],
                         show_progress_every=500000)
                         # show_progress_every=500000, stop_after=1000000)
print('loaded')
print('sorting')
ds.sort(('patient_id', 'updated_at'))
print('sorted')


data_schema = data_schemas.DataSchema(1)
categorical_maps = data_schema.assessment_categorical_maps

results = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
ids = ds.field_by_name('id')
pids = ds.field_by_name('patient_id')
cats = ds.field_by_name('created_at')
uats = ds.field_by_name('updated_at')
hct = ds.field_by_name('had_covid_test')
tcp = ds.field_by_name('tested_covid_positive')
for i_r in range(ds.row_count()):
    results[hct[i_r]][tcp[i_r]] += 1
    if hct[i_r] == 1 and tcp[i_r] == 2:
        print(i_r, ds.field_by_name('patient_id')[i_r])

print('results')
print(results)

if False:
    print('trying schema 1')
    ds.sort(('patient_id', 'updated_at'))
    print('setting up validation test')
    hct_results1 = np.zeros_like(hct, dtype=np.uint8)
    tcp_results1 = np.zeros_like(tcp, dtype=np.uint8)
    filter_status1 = np.zeros(ds.row_count(), dtype=np.uint32)
    fn1 = processing.covid_test.ValidateCovidTestResultsFacVersion1PreHCTFix(hct, tcp, filter_status1, hct_results1, tcp_results1, 0x1,
                                                                             )
                                                             # show_debug=True)
    print('performing validation test')
    utils.iterate_over_patient_assessments(ds.fields_, filter_status1, fn1)

    print('checking results')
    results1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i_r in range(ds.row_count()):
        results1[hct_results1[i_r]][tcp_results1[i_r]] += 1
    print('results')
    print(results1)
    print(np.count_nonzero(filter_status1 == 0))


print('trying schema 1HCTFix')
ds.sort(('patient_id', 'updated_at'))
print('setting up validation test')
hct_results1f = np.zeros_like(hct, dtype=np.uint8)
tcp_results1f = np.zeros_like(tcp, dtype=np.uint8)
filter_status1f = np.zeros(ds.row_count(), dtype=np.uint32)
fn1f = processing.covid_test.ValidateCovidTestResultsFacVersion1(hct, tcp, filter_status1f, hct_results1f, tcp_results1f, 0x1,
                                                                 )
                                                                 # show_debug=True)
print('performing validation test')
utils.iterate_over_patient_assessments(ds.fields_, filter_status1f, fn1f)


print();
print("checking inconsistent test / test results fields")
fn = CheckTestingConsistency(0x2, 0x4)
fn(hct_results1f, tcp_results1f, filter_status1f)
print(
    f'inconsistent_not_tested: filtered {utils.count_flag_set(filter_status1f, 0x2)} missing values')
print(f'inconsistent_tested: filtered {utils.count_flag_set(filter_status1f, 0x4)} missing values')

print('checking results')
results1f = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
for i_r in range(ds.row_count()):
    if filter_status1f[i_r] == 0:
        results1f[hct_results1f[i_r]][tcp_results1f[i_r]] += 1
print('results')
print(results1f)
print(np.count_nonzero(filter_status1f == 0))

# print(np.array_equal(tcp_results1, tcp_results1f))

fn = CalculateMergedFieldCount(ds.field_by_name('updated_at'))
utils.iterate_over_patient_assessments2(pids, filter_status1f, fn)
merged_row_count = ds.row_count() - fn.merged_row_count
merged_fields = {
    'id': [None] * merged_row_count,
    'patient_id': [None] * merged_row_count,
    'created_at': [None] * merged_row_count,
    'updated_at': [None] * merged_row_count,
    'had_covid_test': np.zeros(merged_row_count, dtype=np.uint8),
    'had_covid_test_clean': np.zeros(merged_row_count, dtype=np.uint8),
    'tested_covid_positive': np.zeros(merged_row_count, dtype=np.uint8),
    'tested_covid_positive_clean': np.zeros(merged_row_count, dtype=np.uint8)
}
modified_fields = {
    'had_covid_test': hct,
    'had_covid_test_clean': hct_results1f,
    'tested_covid_positive': tcp,
    'tested_covid_positive_clean': tcp_results1f
}
merged_filter = np.zeros(merged_row_count, dtype=np.uint32)
existing_field_indices = [('id', 0), ('patient_id', 1), ('created_at', 2), ('updated_at', 3)]
merge = MergeAssessmentRows([],
                            merged_fields,
                            modified_fields,
                            existing_field_indices, {},
                            filter_status1f, merged_filter)
utils.iterate_over_patient_assessments(ds.fields_, filter_status1f, merge)
mids = merged_fields['id']
mpids = merged_fields['patient_id']
mcats = merged_fields['created_at']
muats = merged_fields['updated_at']
mhcts = merged_fields['had_covid_test']
mhctc = merged_fields['had_covid_test_clean']
mtcps = merged_fields['tested_covid_positive']
mtcpc = merged_fields['tested_covid_positive_clean']
results1fm = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
anomalous_pids = set()
for i_r in range(merged_row_count):
    if merged_filter[i_r] == 0:
        results1fm[mhctc[i_r]][mtcpc[i_r]] += 1
    if (mhctc[i_r] != 2 and mtcpc[i_r] != 0) or (mhctc[i_r] == 2 and mtcpc[i_r] == 0):
        if merged_filter[i_r] == 0:
            anomalous_pids.add(mpids[i_r])
print(results1fm)

for i_r in range(merged_row_count):
    if mpids[i_r] in anomalous_pids:
        if merged_filter[i_r] == 0:
            print(i_r, mids[i_r], mpids[i_r], mcats[i_r], muats[i_r], mhcts[i_r], mhctc[i_r], mtcps[i_r], mtcpc[i_r])

for i_r in range(ds.row_count()):
    if pids[i_r] in anomalous_pids:
        if filter_status1f[i_r] == 0:
            print(i_r, ids[i_r], pids[i_r], cats[i_r], uats[i_r], hct[i_r], tcp[i_r])


# tcp_index = ds.field_to_index('tested_covid_positive')
# strings_to_values = categorical_maps['tested_covid_positive'].strings_to_values
# patients = defaultdict(int)
# patient_ids = ds.fields_[1]
# tcps = ds.fields_[tcp_index]
# for ir, r in enumerate(patient_ids):
#     patients[r] = max(patients[r], strings_to_values[tcps[ir]])
#
# for ir, r in enumerate(patient_ids):
#     if patients[r] == 0:
#         filter_status[ir] |= 1
#
# remaining_patients = 0
# for k, v in patients.items():
#     if v > 0:
#         remaining_patients += 1
#
# print("patients with 'tested_covid_positive' set: ", remaining_patients)

# print('trying schema 1')
# parsing_schema_1 = parsing_schemas.ParsingSchema(1)
# sanitised_covid_results_1 = np.ndarray((len(ds.fields_),), dtype=np.uint8)
# sanitised_covid_results_key_1 = categorical_maps['tested_covid_positive'].values_to_strings[:]
#
# fn_fac = parsing_schema_1.class_entries['clean_covid_progression']
# fn = fn_fac(ds, filter_status, sanitised_covid_results_key_1, sanitised_covid_results_1, 0x2)
# pipeline.iterate_over_patient_assessments(ds.fields_, filter_status, fn)
#
# print('trying schema 2')
# parsing_schema_2 = parsing_schemas.ParsingSchema(2)
# sanitised_covid_results_2 = np.ndarray((len(ds.fields_),), dtype=np.uint8)
# sanitised_covid_results_key_2 = categorical_maps['tested_covid_positive'].values_to_strings[:]
#
# fn_fac = parsing_schema_2.class_entries['clean_covid_progression']
# fn = fn_fac(ds, filter_status, sanitised_covid_results_key_2, sanitised_covid_results_2, 0x4, show_debug=True)
# pipeline.iterate_over_patient_assessments(ds.fields_, filter_status, fn)
#
# print('unfiltered:', np.count_nonzero(filter_status == 0))
# print(pipeline.build_histogram_from_list(filter_status))
#
# print('reporting')
# patients_flagged = defaultdict(int)
# patient_ids = ds.fields_[1]
# for ir, r in enumerate(patient_ids):
#     patients_flagged[r] |= filter_status[ir]
#
# print(pipeline.build_histogram_from_list(filter_status))
#
# neither = 0
# filtered_by_1 = 0
# filtered_by_2 = 0
# filtered_by_both = 0
# for k, v in patients_flagged.items():
#     if not v:
#         neither += 1
#     else:
#         if v == 0x2:
#             filtered_by_1 += 1
#         if v == 0x4:
#             filtered_by_2 += 1
#         if v == 0x6:
#             filtered_by_both += 1
#
# print('filtered_by_neither:', neither)
# print('filtered_by_1:', filtered_by_1)
# print('filtered_by_2:', filtered_by_2)
# print('filtered_by_both:', filtered_by_both)


