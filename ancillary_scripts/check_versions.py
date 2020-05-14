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

import dataset
import pipeline
import utils

patients_filename = '/home/ben/covid/patients_export_geocodes_20200423050002.csv'
assessments_filename = '/home/ben/covid/assessments_export_20200423050002.csv'
#fn = '/home/ben/covid/assessments_short.csv'
print(f'loading {patients_filename}')
with open(patients_filename) as f:
    ds = dataset.Dataset(f, show_progress_every=500000)

print(utils.build_histogram(ds.field_by_name('version')))

print(f'loading {assessments_filename}')
with open(assessments_filename) as f:
    ds = dataset.Dataset(f, show_progress_every=500000)

print(utils.build_histogram(ds.field_by_name('version')))