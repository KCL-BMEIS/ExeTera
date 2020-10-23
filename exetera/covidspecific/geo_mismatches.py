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

from exetera.core import dataset, utils

# fn1 = '/home/ben/covid/patients_export_geocodes_20200406050002.csv'
# fn2 = '/home/ben/covid/patients_export_geocodes_20200413050002.csv'

fn1 = '/home/ben/covid/patients_export_geocodes_20200413050002.csv'
fn2 = '/home/ben/covid/patients_export_geocodes_20200416050002.csv'

print('loading file 1')
with open(fn1) as f1:
    ds1 = dataset.Dataset(f1)
ds1.sort(('id',))
print('done')

print(); print('loading_file 2')
with open(fn2) as f2:
    ds2 = dataset.Dataset(f2)
ds2.sort(('id',))
print('done')

fields = ('id', 'updated_at', 'year_of_birth', 'height_cm', 'zipcode', 'outward_postcode')
i = 0
j = 0
matches = 0
mismatches = 0
empty = 0
zip1 = ds1.field_to_index('zipcode')
zip2 = ds2.field_to_index('zipcode')
owp1 = ds1.field_to_index('outward_postcode')
owp2 = ds2.field_to_index('outward_postcode')
while i < ds1.row_count() and j < ds2.row_count():
    if ds1.value(i, 0) < ds2.value(j, 0):
        i += 1
    elif ds1.value(i, 0) > ds2.value(j, 0):
        j += 1
    else:
        if i < 100:
            utils.print_diagnostic_row(f'{matches} {i}:', ds1, ds1.fields_, i, fields)
            utils.print_diagnostic_row(f'{matches} {j}:', ds2, ds2.fields_, j, fields)
        vzip1 = ds1.value(i, zip1)
        vzip2 = ds2.value(j, zip2)
        vowp1 = ds1.value(i, owp1)
        vowp2 = ds2.value(j, owp2)
        if vzip1 == '' and vzip2 == '' and vowp1 == '' and vowp2 == '':
            empty += 1
        elif vzip1 == vzip2 and vowp1 == vowp2:
            matches += 1
        else:
            mismatches += 1

        if i > 0 and i % 100000 == 0:
            print(f'{i} {matches}, {mismatches}, {empty}')
        i += 1
        j += 1

print('empty:', empty)
print('matches:', matches)
print('mismatches:', mismatches)