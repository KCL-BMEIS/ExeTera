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

    sort_patients = True
    year_from_age = True
    weight_height_bmi = True

    sort_assessments = True
    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(dataset['assessments'].keys())

    if sort_patients:
        sort_keys = ('id',)
        print(f"sort patients by {sort_keys}")
        id_reader = persistence.NewFixedStringReader(patients_src['id'])
        t1 = time.time()
        sorted_index = persistence.dataset_sort(
            np.arange(len(id_reader), dtype=np.uint32), (id_reader,))
        sorted_ids = persistence.apply_sort_to_array(sorted_index, id_reader[:])
        print(f'sorted {sort_keys} index in {time.time() - t1}s')
        t0 = time.time()
        for k in patients_src.keys():
            t1 = time.time()
            r = persistence.get_reader_from_field(patients_src[k])
            w = r.getwriter(patients_dest, k, timestamp)
            persistence.apply_sort(sorted_index, r, w)
            print(f"'{k}' reordered in {time.time() - t1}s")
        print(f"patient fields reordered in {time.time() - t0}s")

    if year_from_age:
        log("year of birth -> age; 18 to 90 filter")
        t0 = time.time()
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


    if sort_assessments:
        sort_keys = ('patient_id', 'created_at')
        print(f"sort assessments by {sort_keys}")
        t0 = time.time()
        patient_id_reader = persistence.NewFixedStringReader(assessments_src['patient_id'])
        raw_patient_ids = patient_id_reader[:]
        created_at_reader = persistence.NewTimestampReader(assessments_src['created_at'])
        raw_created_ats = created_at_reader[:]
        t1 = time.time()
        sorted_index = persistence.dataset_sort(
            np.arange(len(raw_patient_ids), dtype=np.uint32),
            (patient_id_reader, created_at_reader))
        print(f'sorted {sort_keys} index in {time.time() - t1}s')

        t0 = time.time()
        for k in assessments_src.keys():
            t1 = time.time()
            r = persistence.get_reader_from_field(assessments_src[k])
            w = r.getwriter(assessments_dest, k, timestamp)
            persistence.apply_sort(sorted_index, r, w)
            print(f"'{k}' reordered in {time.time() - t1}s")
        print(f"patient fields reordered in {time.time() - t0}s")

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
