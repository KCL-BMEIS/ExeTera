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
import processing.weight_height_bmi
import regression
import utils
from processing.age_from_year_of_birth import CalculateAgeFromYearOfBirth

def value_ranges():
    lbs_to_kgs = 0.453592
    stones_to_kgs = 6.35029
    cur_min_weight = 40
    cur_max_weight = 200
    cur_min_weight_lbs = cur_min_weight / lbs_to_kgs
    cur_max_weight_lbs = cur_max_weight / lbs_to_kgs
    cur_min_weight = cur_min_weight
    cur_max_weight = cur_max_weight
    cur_min_weight_stones = cur_min_weight / stones_to_kgs
    cur_max_weight_stones = cur_max_weight / stones_to_kgs

    print('lbs:', cur_min_weight_lbs, cur_max_weight_lbs)
    print('kgs:', cur_min_weight, cur_max_weight)
    print('sts:', cur_min_weight_stones, cur_max_weight_stones)

    m_to_cm = 100
    ft_to_cm = 30.48
    in_to_cm = 2.55
    mm_to_cm = 0.1
    cur_min_height = 110
    cur_max_height = 220
    cur_min_height_m = cur_min_height / m_to_cm
    cur_max_height_m = cur_max_height / m_to_cm
    cur_min_height_ft = cur_min_height / ft_to_cm
    cur_max_height_ft = cur_max_height / ft_to_cm
    cur_min_height_in = cur_min_height / in_to_cm
    cur_max_height_in = cur_max_height / in_to_cm
    cur_min_height_mm = cur_min_height / mm_to_cm
    cur_max_height_mm = cur_max_height / mm_to_cm

    print('m:', cur_min_height_m, cur_max_height_m)
    print('ft:', cur_min_height_ft, cur_max_height_ft)
    print('in:', cur_min_height_in, cur_max_height_in)
    print('cm:', cur_min_height, cur_max_height)
    print('mm:', cur_min_height_mm, cur_max_height_mm)

value_ranges()

filename = '/home/ben/covid/patients_export_geocodes_20200428050002.csv'
# filename = '/home/ben/covid/patients_short.csv'

print(f"loading {filename}")
with open(filename) as f:
    ds = dataset.Dataset(f, keys=['id', 'created_at', 'updated_at', 'gender', 'height_cm', 'weight_kg', 'bmi', 'year_of_birth'],
                         # show_progress_every=500000)
                         show_progress_every=500000, stop_after=100000)
print('loaded')

#ds.sort(('patient_id', 'updated_at'))

filter_status = np.zeros(ds.row_count(), dtype=np.uint32)

data_schema = data_schemas.DataSchema(1)



src_genders = ds.field_by_name('gender')
src_yobs = ds.field_by_name('year_of_birth')
src_weights = ds.field_by_name('weight_kg')
src_heights = ds.field_by_name('height_cm')
src_bmis = ds.field_by_name('bmi')


# ages
age_fn = CalculateAgeFromYearOfBirth(0x1, 0x2, utils.valid_range_fac_inc(0, 90), 2020)
ages = np.zeros(len(src_yobs), dtype=np.uint32)
age_fn(src_yobs, ages, filter_status)

print(sorted(utils.build_histogram(ages)))

# gender check
print('gender:', utils.build_histogram(src_genders))


cast_weights = [-1.0 if v == '' else float(v) for v in src_weights]
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 20, 30, 32, 34, 36, 38, 40, 50, 60, 70, 80, 90,
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
        200, 202, 204, 206, 208, 210, 212, 214, 216, 218,
        220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360, 380,
        400, 450, 500, 600, 700, 800, 900, 1000, 10000, 100000, 100000, 1000000, 100000000]
cast_weights_hist = np.histogram(cast_weights, bins=bins)
for k, v in zip(cast_weights_hist[1], cast_weights_hist[0]):
    print(f'{k}: {v}')
print('min ; max =', min(cast_weights), max(cast_weights))

results = [0, 0, 0, 0]
ambiguous = list()
for i_r in range(len(src_weights)):
    value = 0
    if src_weights[i_r] != '':
        value += 1
    if src_heights[i_r] != '':
        value += 2
    if value == 1 or value == 2:
        print(i_r, regression.na_or_value(src_weights[i_r]), regression.na_or_value(src_heights[i_r]))
    results[value] += 1

print(results)


class ValueRange:
    def __init__(self):
        self.min = None
        self.max = None
        self.count = 0

    def add_value(self, value):
        if self.min is None:
            self.min = value
        else:
            self.min = min(self.min, value)
        if self.max is None:
            self.max = value
        else:
            self.max = max(self.max, value)
        self.count += 1

max_height = None
for i_r in range(len(src_heights)):
    if src_heights[i_r] != '':
        fheight = float(src_heights[i_r])
        if max_height is None or fheight > max_height:
            max_height = fheight
            print(i_r, src_heights[i_r])

float_heights = np.zeros_like(src_heights, dtype=np.float)
for i_r in range(len(src_heights)):
    float_heights[i_r] = float(src_heights[i_r]) if src_heights[i_r] is not '' else 0.0

sorted_float_heights = np.sort(float_heights)
print(sorted_float_heights[-100:])
print(np.percentile(float_heights, [0, 1, 5, 10, 50, 90, 95, 99, 100]))

created_at = ds.field_by_name('created_at')
updated_at = ds.field_by_name('updated_at')
weight_range_by_day = defaultdict(ValueRange)
height_range_by_day = defaultdict(ValueRange)
for i_r in range(len(updated_at)):
    if src_weights[i_r] != '':
        weight_range_by_day[utils.timestamp_to_day(created_at[i_r])].add_value(float(src_weights[i_r]))
    if src_heights[i_r] != '':
        height_range_by_day[utils.timestamp_to_day(created_at[i_r])].add_value(float(src_heights[i_r]))

weight_ranges = sorted(weight_range_by_day.items())
height_ranges = sorted(height_range_by_day.items())
for w in weight_ranges:
    print(w[0], w[1].count, w[1].min, w[1].max)
for h in height_ranges:
    print(h[0], h[1].count, h[1].min, h[1].max)

compare_schemas = False
if compare_schemas:
    parsing_schema_1 = parsing_schemas.ParsingSchema(1)

    fn_fac_1 = parsing_schema_1.class_entries['validate_weight_height_bmi']
    fn_1 = fn_fac_1(pipeline.MIN_WEIGHT, pipeline.MAX_WEIGHT,
                    pipeline.MIN_HEIGHT, pipeline.MAX_HEIGHT,
                    pipeline.MIN_BMI, pipeline.MAX_BMI,
                    0x1, 0x4, 0x10, 0x40, 0x100, 0x400)

    weight_clean_1, height_clean_1, bmi_clean_1 =\
        fn_1(src_genders, ages, src_weights, src_heights, src_bmis, filter_status)

    parsing_schema_2 = parsing_schemas.ParsingSchema(2)

    fn_fac_2 = parsing_schema_2.class_entries['validate_weight_height_bmi']
    fn_2 = fn_fac_2(pipeline.MIN_WEIGHT, pipeline.MAX_WEIGHT,
                    pipeline.MIN_HEIGHT, pipeline.MAX_HEIGHT,
                    pipeline.MIN_BMI, pipeline.MAX_BMI,
                    0x2, 0x8, 0x20, 0x80, 0x200, 0x800)

    weight_clean_2, height_clean_2, bmi_clean_2 =\
        fn_2(src_genders, ages, src_weights, src_heights, src_bmis, filter_status)

    histogram = utils.build_histogram(filter_status)
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
else:
    fn_fac_3 = processing.weight_height_bmi.ValidateHeight3
    fn_3 = fn_fac_3(pipeline.MIN_WEIGHT, pipeline.MAX_WEIGHT,
                    pipeline.MIN_HEIGHT, pipeline.MAX_HEIGHT,
                    pipeline.MIN_BMI, pipeline.MAX_BMI,
                    0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80)

    weight_clean_3, height_clean_3, bmi_clean_3 = \
        fn_3(src_genders, ages, src_weights, src_heights, src_bmis, filter_status)

# for i in range(1000):
#     r = ds.fields_[i]
#     print(i, r[0], r[ds.field_to_index('weight_kg')], weight_clean_2[i],
#           r[ds.field_to_index('height_cm')], height_clean_2[i],
#           r[ds.field_to_index('bmi')], bmi_clean_2[i])
