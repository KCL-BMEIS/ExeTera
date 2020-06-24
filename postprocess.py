from datetime import datetime, timezone
import time
import math

import numpy as np
import h5py
from numba import jit, njit

import utils
from processing.age_from_year_of_birth import calculate_age_from_year_of_birth_fast
from processing.weight_height_bmi import weight_height_bmi_fast_1
from processing.inconsistent_symptoms import check_inconsistent_symptoms_1
import data_schemas
import parsing_schemas
import persistence


# TODO: base filter for all hard filtered things, or should they be blitzed
# from the dataset completely?

# TODO: postprocessing activities
# * assessment sort by (patient_id, created_at)
# * aggregate from assessments to patients
#   * was first unwell
#   * first assessment
#   * last assessment
#   * assessment count
#   * assessment index start
#   * assessment index end

def log(*a, **kwa):
    print(*a, **kwa)

def postprocess(dataset, destination, data_schema, process_schema, timestamp=None):

    chunksize = 1 << 20

    patients_src = dataset['patients']
    patients_dest = destination.create_group('patients')
    assessments_src = dataset['assessments']
    assessments_dest = destination.create_group('assessments')

    pre_sort_assessments = True
    year_from_age = True
    weight_height_bmi = True
    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(dataset['assessments'].keys())

    # t0 = time.time()
    # value = 100
    # count = len(patients_src['year_of_birth']['values'])
    # w = persistence.NumericWriter(patients_dest, chunksize, 'stuff', timestamp, 'uint32',
    #                               needs_filter=True)
    # for i in range(count):
    #     w.append(value)
    # w.flush()
    # log(f"completed in {time.time() - t0}")


    if year_from_age:
        log("year of birth -> age; 18 to 90 filter")
        t0 = time.time()
        # calculate_age_from_year_of_birth(
        #     patients_dest, patients_src['year_of_birth'],
        #     utils.valid_range_fac_inc(16, 90), 2020, chunksize, timestamp, name='age')
        age = persistence.NewNumericWriter(patients_dest, chunksize, 'age', timestamp, 'uint32')
        age_filter = persistence.NewNumericWriter(patients_dest, chunksize, 'age_filter',
                                                  timestamp, 'bool')
        age_16_to_90 = persistence.NewNumericWriter(patients_dest, chunksize, '16_to_90_years',
                                                timestamp, 'bool')
        calculate_age_from_year_of_birth_fast(
            16, 90,
            patients_src['year_of_birth'], patients_src['year_of_birth_valid'],
            age, age_filter, age_16_to_90,
            2020)
        log(f"completed in {time.time() - t0}")

        print('age_filter count:', np.sum(patients_dest['age_filter']['values'][:]))
        print('16_to_90_years count:', np.sum(patients_dest['16_to_90_years']['values'][:]))

    if weight_height_bmi:
        log("height / weight / bmi; standard range filters")
        t0 = time.time()

        weights_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'weight_kg_clean',
                                                     timestamp, 'float32')
        weights_filter = persistence.NewNumericWriter(patients_dest, chunksize, '40_to_200_kg',
                                                      timestamp, 'bool')
        heights_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'height_cm_clean',
                                                     timestamp, 'float32')
        heights_filter = persistence.NewNumericWriter(patients_dest, chunksize, '110_to_220_cm',
                                                      timestamp, 'bool')
        bmis_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'bmi_clean',
                                                  timestamp, 'float32')
        bmis_filter = persistence.NewNumericWriter(patients_dest, chunksize, '15_to_55_bmi',
                                                   timestamp, 'bool')

        weight_height_bmi_fast_1(40, 200, 110, 220, 15, 55,
                                 None, None, None, None,
                                 patients_src['weight_kg'], patients_src['weight_kg_valid'],
                                 patients_src['height_cm'], patients_src['height_cm_valid'],
                                 patients_src['bmi'], patients_src['bmi_valid'],
                                 weights_clean, weights_filter, None,
                                 heights_clean, heights_filter, None,
                                 bmis_clean, bmis_filter, None)
        log(f"completed in {time.time() - t0}")


    sort_keys = ('patient_id', 'created_at')
    print(f"sort assessments by {sort_keys}")
    t0 = time.time()
    patient_id_reader = persistence.NewFixedStringReader(assessments_src['patient_id'])
    raw_patient_ids = patient_id_reader[:]
    created_at_reader = persistence.NewTimestampReader(assessments_src['created_at'])
    raw_created_ats = created_at_reader[:]
    t1 = time.time()
    sorted_index = persistence.dataset_sort(
        np.arange(len(raw_patient_ids)), (raw_patient_ids, raw_created_ats))
    print(f'sorted {sort_keys} index in {time.time() - t1}s')
    t1 = time.time()

    result_patient_ids = persistence.apply_sort(sorted_index, raw_patient_ids)
    result_created_ats = persistence.apply_sort(sorted_index, raw_created_ats)

    print(f'reindexed_keys in {time.time() - t1}s')
    persistence.NewFixedStringWriter(assessments_dest, patient_id_reader.chunksize,
                                     'patient_id', timestamp, patient_id_reader.dtype())
    print(f'completed in {time.time() - t0}s')

    print('check inconsistent health_status')
    t0 = time.time()
    check_inconsistent_symptoms_1(assessments_src, assessments_dest, timestamp)
    print(time.time() - t0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='the dataset to load')
    parser.add_argument('-t', '--temporary', required=False, default=None,
                        help='a temporary dataset to write results to')
    args = parser.parse_args()

    data_schema = data_schemas.DataSchema(1)
    parsing_schema = parsing_schemas.ParsingSchema(1)
    timestamp = str(datetime.now(timezone.utc))

    if args.temporary is None:
        tempfilename = persistence.temp_filename()
    else:
        tempfilename = args.temporary
    with h5py.File(args.dataset, 'r') as ds:
        with h5py.File(tempfilename, 'w') as ts:
            postprocess(ds, ts, data_schema, parsing_schema, timestamp)
