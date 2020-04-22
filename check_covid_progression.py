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
filename = '/home/ben/covid/assessments_export_20200421050002.csv'

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
    ds = dataset.Dataset(f, progress)#, stop_after=5000000)
print('loaded')

ds.sort(('patient_id', 'updated_at'))

filter_status = np.zeros(len(ds.fields_,), dtype=np.uint32)

data_schema = data_schemas.DataSchema(1)
categorical_maps = data_schema.get_assessment_categorical_maps()

tcp_index = ds.field_to_index('tested_covid_positive')
strings_to_values = categorical_maps['tested_covid_positive'].strings_to_values
patients = defaultdict(int)
for ir, r in enumerate(ds.fields_):
    patients[r[1]] = max(patients[r[1]], strings_to_values[r[tcp_index]])

for ir, r in enumerate(ds.fields_):
    if patients[r[1]] == 0:
        filter_status[ir] |= 1

remaining_patients = 0
for k, v in patients.items():
    if v > 0:
        remaining_patients += 1

print("patients with 'tested_covid_positive' set: ", remaining_patients)

print('trying schema 1')
parsing_schema_1 = parsing_schemas.ParsingSchema(1)
sanitised_covid_results_1 = np.ndarray((len(ds.fields_),), dtype=np.uint8)
sanitised_covid_results_key_1 = categorical_maps['tested_covid_positive'].values_to_strings[:]

fn_fac = parsing_schema_1.class_entries['clean_covid_progression']
fn = fn_fac(ds, filter_status, sanitised_covid_results_key_1, sanitised_covid_results_1, 0x2)
pipeline.iterate_over_patient_assessments(ds.fields_, filter_status, fn)

print('trying schema 2')
parsing_schema_2 = parsing_schemas.ParsingSchema(2)
sanitised_covid_results_2 = np.ndarray((len(ds.fields_),), dtype=np.uint8)
sanitised_covid_results_key_2 = categorical_maps['tested_covid_positive'].values_to_strings[:]

fn_fac = parsing_schema_2.class_entries['clean_covid_progression']
fn = fn_fac(ds, filter_status, sanitised_covid_results_key_2, sanitised_covid_results_2, 0x4, show_debug=True)
pipeline.iterate_over_patient_assessments(ds.fields_, filter_status, fn)

print('unfiltered:', np.count_nonzero(filter_status == 0))
print(pipeline.build_histogram_from_list(filter_status))

print('reporting')
patients_flagged = defaultdict(int)
for ir, r in enumerate(ds.fields_):
    patients_flagged[r[1]] |= filter_status[ir]

print(pipeline.build_histogram_from_list(filter_status))

neither = 0
filtered_by_1 = 0
filtered_by_2 = 0
filtered_by_both = 0
for k, v in patients_flagged.items():
    if not v:
        neither += 1
    else:
        if v == 0x2:
            filtered_by_1 += 1
        if v == 0x4:
            filtered_by_2 += 1
        if v == 0x6:
            filtered_by_both += 1

print('filtered_by_neither:', neither)
print('filtered_by_1:', filtered_by_1)
print('filtered_by_2:', filtered_by_2)
print('filtered_by_both:', filtered_by_both)


