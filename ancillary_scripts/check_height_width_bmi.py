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

import numpy as np

import dataset
import data_schemas
import parsing_schemas
import pipeline

filename = '/home/ben/covid/patients_export_geocodes_20200421050002.csv'
# filename = '/home/ben/covid/patients_short.csv'

print(f"loading {filename}")
with open(filename) as f:
    ds = dataset.Dataset(f, progress=True, stop_after=1000000)
print('loaded')

#ds.sort(('patient_id', 'updated_at'))

filter_status = np.zeros(ds.row_count(), dtype=np.uint32)

data_schema = data_schemas.DataSchema(1)


yob = ds.field_by_name('year_of_birth')
for ir in range(ds.row_count()):
    if yob[ir] == '' or\
       int(float(yob[ir])) < pipeline.MIN_YOB or int(float(yob[ir])) > pipeline.MAX_YOB:
        filter_status[ir] |= 0x1000

src_weights = ds.field_by_name('weight_kg')
src_heights = ds.field_by_name('height_cm')
src_bmis = ds.field_by_name('bmi')


parsing_schema_1 = parsing_schemas.ParsingSchema(1)

fn_fac_1 = parsing_schema_1.class_entries['validate_weight_height_bmi']
fn_1 = fn_fac_1(pipeline.MIN_WEIGHT, pipeline.MAX_WEIGHT,
                pipeline.MIN_HEIGHT, pipeline.MAX_HEIGHT,
                pipeline.MIN_BMI, pipeline.MAX_BMI,
                0x1, 0x4, 0x10, 0x40, 0x100, 0x400)

weight_clean_1, height_clean_1, bmi_clean_1 =\
    fn_1(src_weights, src_heights, src_bmis, filter_status)

parsing_schema_2 = parsing_schemas.ParsingSchema(2)

fn_fac_2 = parsing_schema_2.class_entries['validate_weight_height_bmi']
fn_2 = fn_fac_2(pipeline.MIN_WEIGHT, pipeline.MAX_WEIGHT,
                pipeline.MIN_HEIGHT, pipeline.MAX_HEIGHT,
                pipeline.MIN_BMI, pipeline.MAX_BMI,
                0x2, 0x8, 0x20, 0x80, 0x200, 0x800)

weight_clean_2, height_clean_2, bmi_clean_2 =\
    fn_2(src_weights, src_heights, src_bmis, filter_status)

histogram = pipeline.build_histogram_from_list(filter_status)
histogram = sorted(histogram, key=lambda x: x[0])

print('no flag,', 'first,', 'second,', 'both')
bad_weight = [0, 0, 0, 0]
for i_f, f in enumerate(filter_status):
    if f & 0x1 == 0:
        bad_weight[(f & 0xc) >> 2] += 1
print(bad_weight)

bad_height = [0, 0, 0, 0]
for f in filter_status:
    if f & 0x10 == 0:
        bad_height[(f & 0xc0) >> 6] += 1
print(bad_height)

bad_bmi = [0, 0, 0, 0]
for f in filter_status:
    if f & 0x100 == 0:
        bad_bmi[(f & 0xc00) >> 10] += 1
print(bad_bmi)

print(np.count_nonzero(filter_status & 0x8 != 0))


# for i in range(1000):
#     r = ds.fields_[i]
#     print(i, r[0], r[ds.field_to_index('weight_kg')], weight_clean_2[i],
#           r[ds.field_to_index('height_cm')], height_clean_2[i],
#           r[ds.field_to_index('bmi')], bmi_clean_2[i])
