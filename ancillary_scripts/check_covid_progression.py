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

#filename = '/home/ben/covid/assessments_short.csv'
filename = '/home/ben/covid/assessments_export_20200423050002.csv'
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
                         keys=['id', 'patient_id', 'updated_at', 'had_covid_test', 'tested_covid_positive'],
                         progress=True)
                         # progress = True, stop_after = 1000000)
print('loaded')


data_schema = data_schemas.DataSchema(1)
categorical_maps = data_schema.assessment_categorical_maps

results = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
hct = ds.field_by_name('had_covid_test')
tcp = ds.field_by_name('tested_covid_positive')
for i_r in range(ds.row_count()):
    results[hct[i_r]][tcp[i_r]] += 1
    if hct[i_r] == 1 and tcp[i_r] == 2:
        print(i_r, ds.field_by_name('patient_id')[i_r])

print('results')
print(results)

print('trying schema 1')
ds.sort(('patient_id', 'updated_at'))
print('setting up validation test')
hct_results = np.zeros_like(hct, dtype=np.uint8)
tcp_results = np.zeros_like(tcp, dtype=np.uint8)
filter_status = np.zeros(ds.row_count(), dtype=np.uint32)
fn = parsing_schemas.ValidateCovidTestResultsFacVersion1(hct, tcp, filter_status, None, hct_results, tcp_results, 0x1,
                                                         show_debug=True)
print('performing validation text')
pipeline.iterate_over_patient_assessments(ds.fields_, filter_status, fn)

print('checking results')
results = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
for i_r in range(ds.row_count()):
    results[hct_results[i_r]][tcp_results[i_r]] += 1
print('results')
print(results)
print(np.count_nonzero(filter_status == 0))

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


