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
import regression

import numpy as np


# filename1 = 'v0.1.7_50k_patients.csv'
# filename2 = 'test_patients.csv'
# sort_keys = ('id',)
# diagnostic_keys = ['id', 'weight_kg', 'height_cm', 'weight_clean', 'height_clean']
filename1 = '/home/ben/git/zoe-data-prep-stable/v0.1.7_assessments_0413_full_out.csv'
filename2 = '/home/ben/git/zoe-data-prep-stable/v0.1.8_assessments_0413_full_out.csv'
sort_keys = ('patient_id', 'updated_at')
diagnostic_keys = [s for s in sort_keys]
keys_to_compare = ['health_status', 'fatigue', 'fatigue_binary', ('had_covid_test', 'had_covid_test_clean'), 'tested_covid_positive', ('tested_covid_positive', 'tested_covid_positive_clean')]

with open(filename1) as f:
    ds1 = dataset.Dataset(f, progress=True, stop_after=99999)

with open(filename2) as f:
    ds2 = dataset.Dataset(f, progress=True, stop_after=99999)


print(ds1.row_count())
ds1.sort(sort_keys)
print(ds2.row_count())
ds2.sort(sort_keys)

fields = set(ds1.names_).intersection(set(ds2.names_))
print(fields)

def match_rows(k1, k2):
    x = 0
    y = 0
    xindices = list()
    yindices = list()
    while x < len(k1) and y < len(k2):
        if k1[x] < k2[y]:
            x += 1
        elif k1[x] > k2[y]:
            y += 1
        else:
            xindices.append(x)
            yindices.append(y)
            x += 1
            y += 1
    return xindices, yindices

def elements_not_equal(xinds, yinds, f1, f2):
    discrepencies = None
    for r in range(len(xinds)):
        x = xinds[r]
        y = yinds[r]
        if f1[x] != f2[y]:
            if discrepencies is None:
                discrepencies = list()
            discrepencies.append(r)

    return discrepencies

k1 = ds1.field_by_name('id')
k2 = ds2.field_by_name('id')
xinds, yinds = match_rows(k1, k2)

print('hct -> hct:', np.array_equal(ds1.field_by_name('had_covid_test'), ds2.field_by_name('had_covid_test')))
print('hct -> hctc:', np.array_equal(ds1.field_by_name('had_covid_test'), ds2.field_by_name('had_covid_test_clean')))
print('tcp -> tcp:', np.array_equal(ds1.field_by_name('tested_covid_positive'), ds2.field_by_name('tested_covid_positive')))
print('tcp -> tcpc:', np.array_equal(ds1.field_by_name('tested_covid_positive'), ds2.field_by_name('tested_covid_positive_clean')))

for i in range(len(xinds)):
    x = xinds[i]
    y = yinds[i]
    disparities = regression.check_row(ds1, x, ds2, y, keys_to_compare, dict())
    if disparities is not None:
        print(x, y, disparities)

# for f in fields:
#
#     f1 = ds1.field_by_name(f)
#     f2 = ds2.field_by_name(f)
#     discrepencies = elements_not_equal(xinds, yinds, f1, f2)
#     if discrepencies is not None:
#         for d in discrepencies:
#             v1 = ds1.value_from_fieldname(xinds[d], f)
#             v2 = ds2.value_from_fieldname(yinds[d], f)
#             print(xinds[d], yinds[d],
#                   'na' if v1 == '' else v1, '|', 'na' if v2 == '' else v2)
#             pipeline.print_diagnostic_row(xinds[d], ds1, xinds[d], diagnostic_keys + [f])
#             pipeline.print_diagnostic_row(yinds[d], ds2, yinds[d], diagnostic_keys + [f])

#
# for r in range(ds1.row_count()):
#     for f in fields:
#         v1 = ds1.value_from_fieldname(r, f)
#         v2 = ds2.value_from_fieldname(r, f)
#         if v1 != v2:
#             print('discrepency:', r, f, v1, v2)
#
# is_carer_for_community = ds1.field_by_name('is_carer_for_community')
# print(is_carer_for_community.count(''))
#
# for n in ds2.names_:
#     if elements_equal(is_carer_for_community, ds2.field_by_name(n)):
#         print('match with ', n)
