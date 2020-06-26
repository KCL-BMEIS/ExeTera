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
from processing.temperature import validate_temperature_1
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
    daily_assessments_dest = destination.create_group('daily_assessments')
    tests_src = dataset['tests']
    tests_dest = destination.create_group('tests')

    sort_patients = True
    year_from_age = True
    clean_weight_height_bmi = True

    sort_assessments = True
    clean_temperatures = True

    sort_tests = True

    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(dataset['assessments'].keys())

    # Patient processing
    # ==================

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
            del r
            del w
            print(f"  '{k}' reordered in {time.time() - t1}s")
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

    if clean_weight_height_bmi:
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

    # Assessment processing
    # =====================

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
            del r
            del w
            print(f"  '{k}' reordered in {time.time() - t1}s")
        print(f"assessment fields reordered in {time.time() - t0}s")

        print("checking sort order")
        t0 = time.time()
        raw_patient_ids = persistence.NewFixedStringReader(assessments_dest['patient_id'])[:]
        raw_created_ats = persistence.NewTimestampReader(assessments_dest['created_at'])[:]
        last_pid = raw_patient_ids[0]
        last_cat = raw_created_ats[0]
        duplicates = 0
        for i_r in range(1, len(patient_id_reader)):
            pid = raw_patient_ids[i_r]
            cat = raw_created_ats[i_r]
            if (last_pid, last_cat) == (pid, cat):
                duplicates += 1
            if (last_pid, last_cat) > (pid, cat):
                print(i_r,
                      last_pid, datetime.fromtimestamp(last_cat),
                      pid, datetime.fromtimestamp(cat))
            last_pid = pid
            last_cat = cat
            # if i_r < 1000:
            #     print(i_r, pid, datetime.fromtimestamp(cat))
        print(f"sort order checked({duplicates} duplicate row keys found) in {time.time() - t0}")

    # if full_assessment_valid:


    if clean_temperatures:
        print("clean temperatures")
        t0 = time.time()
        temps = persistence.NewNumericReader(assessments_dest['temperature'])
        temp_units = persistence.NewFixedStringReader(assessments_dest['temperature_unit'])
        temps_valid = persistence.NewNumericReader(assessments_dest['temperature_valid'])
        dest_temps = temps.getwriter(assessments_dest, 'temperature_c_clean', timestamp)
        dest_temps_valid =\
            temps_valid.getwriter(assessments_dest, 'temperature_35_to_42_inclusive', timestamp)
        dest_temps_modified =\
            temps_valid.getwriter(assessments_dest, 'temperature_modified', timestamp)
        validate_temperature_1(35.0, 42.0,
                               temps, temp_units, temps_valid,
                               dest_temps, dest_temps_valid, dest_temps_modified)
        print(f"temperature cleaning done in {time.time() - t0}")


    print('check inconsistent health_status')
    t0 = time.time()
    check_inconsistent_symptoms_1(assessments_src, assessments_dest, timestamp)
    print(time.time() - t0)


    # Daily assessments
    # =================

    spids = set()
    patient_ids = persistence.get_reader_from_field(assessments_src['patient_id'])
    utils.iterate_over_patient_assessments2(
        patient_ids[:], np.ones(len(patient_ids)), lambda curid,_1,_2,_3,: spids.add(curid)
    )


    # Test processing
    # ===============

    if sort_tests:
        sort_keys = ('patient_id', 'created_at')
        print(f"sort tests by {sort_keys}")
        t0 = time.time()
        patient_id_reader = persistence.NewFixedStringReader(tests_src['patient_id'])
        raw_patient_ids = patient_id_reader[:]
        created_at_reader = persistence.NewTimestampReader(tests_src['created_at'])
        raw_created_ats = created_at_reader[:]
        t1 = time.time()
        sorted_index = persistence.dataset_sort(
            np.arange(len(raw_patient_ids), dtype=np.uint32),
            (patient_id_reader, created_at_reader))
        print(f'sorted {sort_keys} index in {time.time() - t1}s')

        t0 = time.time()
        for k in tests_src.keys():
            t1 = time.time()
            r = persistence.get_reader_from_field(tests_src[k])
            w = r.getwriter(tests_dest, k, timestamp)
            persistence.apply_sort(sorted_index, r, w)
            del r
            del w
            print(f"  '{k}' reordered in {time.time() - t1}s")
        print(f"test fields reordered in {time.time() - t0}s")


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
