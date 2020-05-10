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
from utils import build_histogram

patient_filename = '/home/ben/covid/patients_export_geocodes_20200504030002.csv'

core_keys = ('id', 'created_at', 'updated_at')
past_symptom_keys =\
    ('still_have_past_symptoms',
     'past_symptoms_days_ago', 'past_symptoms_changed',
     'past_symptom_anosmia', 'past_symptom_shortness_of_breath',
     'past_symptom_fatigue', 'past_symptom_fever',
     'past_symptom_skipped_meals', 'past_symptom_persistent_cough',
     'past_symptom_diarrhoea', 'past_symptom_chest_pain',
     'past_symptom_hoarse_voice', 'past_symptom_abdominal_pain',
     'past_symptom_delirium')

with open(patient_filename) as f:
    ds = dataset.Dataset(f, keys=core_keys + past_symptom_keys,
                         progress=True)
                         # progress=True, stop_after=999999)
    ds.sort(('created_at', 'id'))

for p in past_symptom_keys:
    field = ds.field_by_name(p)
    if p == 'past_symptoms_days_ago':
        histogram = build_histogram(field)
        nones = None
        for h in histogram:
            if h[0] is '':
                nones = h[1]
        histogram = [(int(v[0]), v[1])
                     for v in build_histogram(field) if v[0] is not '']
        if nones is not None:
            histogram = [(None, nones)] + sorted(histogram)
        else:
            histogram = sorted(histogram)
        print(f"{p}:", histogram)
    else:
        print(f"{p}:", sorted(build_histogram(field)))

p_ids = ds.field_by_name('id')
p_c_ats = ds.field_by_name('created_at')
p_u_ats = ds.field_by_name('updated_at')
min_ir = ds.row_count()
for p in past_symptom_keys:
    p_symp = ds.field_by_name(p)
    for i_r in range(ds.row_count()):
        if p_symp[i_r] != '':
            print(p, i_r, p_symp[i_r])
            min_ir = min(min_ir, i_r)
            break

print(min_ir, p_ids[min_ir], p_c_ats[min_ir], p_u_ats[min_ir])
values = []
for p in past_symptom_keys:
    values.append(str(ds.field_by_name(p)[min_ir]))
print(values)
