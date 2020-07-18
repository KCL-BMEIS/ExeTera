import csv

import numpy as np

from hytable.core import dataset
from hytable.covidspecific import data_schemas
from processing.weight_height_bmi import ValidateHeight2

filename = '/home/ben/covid/patients_export_geocodes_20200701030002.csv'

MIN_AGE = 16
MAX_AGE = 90
MIN_HEIGHT = 110
MAX_HEIGHT = 220
MIN_WEIGHT = 40
MAX_WEIGHT = 200
MIN_BMI = 15
MAX_BMI = 55

keys = ('id', 'weight_kg', 'height_cm', 'bmi')
data_schema = data_schemas.DataSchema(1)
stop_after = None
with open(filename) as f:
    ds = dataset.Dataset(f, data_schema.patient_categorical_maps,
                         keys=keys, show_progress_every=100000, stop_after=stop_after)

vh = ValidateHeight2(MIN_WEIGHT, MAX_WEIGHT, MIN_HEIGHT, MAX_HEIGHT, MIN_BMI, MAX_BMI,
                     0x0, 0x0, 0x1, 0x2, 0x4, 0x8, 0x10, 0x20)
id = ds.field_by_name('id')
weightstr = ds.field_by_name('weight_kg')

# def field_to_float(strfield):
#     floatfield = np.zeros(len(strfield), dtype=np.float32)
#     for i_r in range(len(strfield)):
#         if strfield[i_r] == '':
#             floatfield[i_r] = 0.0
#         else:
#             floatfield[i_r] = strfield[i_r]
#     return floatfield


# weights = field_to_float(ds.field_by_name('weight_kg'))
# heights = field_to_float(ds.field_by_name('height_cm'))
# bmis = field_to_float(ds.field_by_name('bmi'))
weights = ds.field_by_name('weight_kg')
heights = ds.field_by_name('height_cm')
bmis = ds.field_by_name('bmi')
flags = np.zeros(ds.row_count(), dtype=np.uint32)
weight_clean, height_clean, bmi_clean =\
    vh(None, None, weights, heights, bmis, flags)

missing_weight = np.where(flags & 0x1, 1, 0)
bad_weight = np.where(flags & 0x2, 1, 0)
missing_height = np.where(flags & 0x4, 1, 0)
bad_height = np.where(flags & 0x8, 1, 0)
missing_bmi = np.where(flags & 0x10, 1, 0)
bad_bmi = np.where(flags & 0x20, 1, 0)


print('writing results')
with open('results.csv', 'w') as f:
    csvw = csv.writer(f)
    csvw.writerow(keys + ('flags', 'weight_clean', 'height_clean', 'bmi_clean',
                          'missing_weight', 'bad_weight', 'missing_height', 'bad_height',
                          'missing_bmi', 'bad_bmi'))
    for i_r in range(ds.row_count()):
        if i_r % 100000 == 0:
            print(i_r)
        csvw.writerow([id[i_r], weights[i_r], heights[i_r], bmis[i_r],
                       weight_clean[i_r], height_clean[i_r], bmi_clean[i_r],
                       missing_weight[i_r], bad_weight[i_r],
                       missing_height[i_r], bad_height[i_r],
                       missing_bmi[i_r], bad_bmi[i_r]])
    print(i_r)



