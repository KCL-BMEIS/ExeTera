from collections import defaultdict
import numpy as np

from numba import jit
import numba

import dataset
import pipeline
import data_schemas

from utils import count_flag_set


def timed_fn(fn):
    import time
    def _inner(*args, **kwargs):
        ts = time.time()
        result = fn(*args, **kwargs)
        print('elapsed:', time.time() - ts)
        return result
    return _inner


@timed_fn
def check_missing(field, flags, code):
    for i_r in range(len(field)):
        if field[i_r] == '':
            flags[i_r] |= code


def field_to_uint32(values):
    result = np.zeros(len(values), dtype=np.uint32)
    for i_v in range(len(values)):
        result[i_v] = 0 if values[i_v] == '' else int(float(values[i_v]))

    return result


def field_to_float32(values):
    result = np.zeros(len(values), dtype=np.float32)
    for i_v in range(len(values)):
        result[i_v] = 0 if values[i_v] == '' else float(values[i_v])

    return result

# @timed_fn
# @numba.vectorize(['uint32(uint32, uint32, uint32)', 'float32(float32, uint32, uint32)'])
# def check_missing(field, flag, code):
#     return flag | 0 if field == 0 else code

count_flag_set = timed_fn(count_flag_set)


with open('/home/ben/covid/patients_export_geocodes_20200428050002.csv') as f:
    p_ds = dataset.Dataset(f, keys=['id', 'year_of_birth', 'weight_kg', 'height_cm', 'bmi'],
                           progress=True)

p_filter_flags = np.zeros(p_ds.row_count(), dtype=np.uint32)

p_fields_to_check = (('year_of_birth', 0x1), ('weight_kg', 0x2), ('height_cm', 0x4), ('bmi', 0x8))

# cast_functions = {'year_of_birth': field_to_uint32, 'weight_kg': field_to_float32, 'height_cm': field_to_float32,
#                   'bmi': field_to_float32}
# cast_fields = dict()
# for p in p_fields_to_check:
#     cast_fields[p[0]] = cast_functions[p[0]](p_ds.field_by_name(p[0]))
#
# for name, value in p_fields_to_check:
#     check_missing(cast_fields[name], p_filter_flags, value)

for name, value in p_fields_to_check:
    check_missing(p_ds.field_by_name(name), p_filter_flags, value)
    count_default = 0
    for i in p_ds.field_by_name(name):
        if i == '0.0':
            count_default += 1
    print('instances of default value:', count_default)

print(p_ds.row_count())

for name, value in p_fields_to_check:
    print(f'{name}:', count_flag_set(p_filter_flags, value))
print('combined:', count_flag_set(p_filter_flags, 0xffff))

p_ids = p_ds.field_by_name('id')
filtered_patients = set()
for i_f, f in enumerate(p_filter_flags):
    if f != 0:
        filtered_patients.add(p_ids[i_f])

with open('/home/ben/covid/assessments_export_20200428050002.csv') as f:
    a_ds = dataset.Dataset(f, keys=['id', 'patient_id'], progress=True)

a_pids = a_ds.field_by_name('patient_id')

assessment_filter_count = 0
for i_p, p in enumerate(a_pids):
    if p in filtered_patients:
        assessment_filter_count += 1

print('assessments filtered by patient_filtering:', assessment_filter_count)