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

def postprocess(dataset, destination, data_schema, process_schema, timestamp=None, flags='all'):

    chunksize = 1 << 20

    patients_src = dataset['patients']
    # patients_dest = destination.create_group('patients')
    patients_dest = persistence.get_or_create_group(destination, 'patients')
    assessments_src = dataset['assessments']
    # assessments_dest = destination.create_group('assessments')
    assessments_dest = persistence.get_or_create_group(destination, 'assessments')
    daily_assessments_dest = persistence.get_or_create_group(destination, 'daily_assessments')
    tests_src = dataset['tests']
    tests_dest = persistence.get_or_create_group(destination, 'tests')

    sort_enabled = lambda x: x in ('sort', 'all')
    process_enabled = lambda x: x in ('process', 'all')
    sort_patients = sort_enabled(flags) and True
    sort_assessments = sort_enabled(flags) and True
    sort_tests = sort_enabled(flags) and True

    year_from_age = process_enabled(flags) and True
    clean_weight_height_bmi = process_enabled(flags) and True
    clean_temperatures = process_enabled(flags) and True
    check_symptoms = process_enabled(flags) and True
    create_daily = process_enabled(flags) and True

    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())
    print(dataset['assessments'].keys())

    # write_mode = 'overwrite'
    write_mode = 'write'

    # Sorting
    # #######

    if sort_patients:
        sort_keys = ('id',)
        persistence.sort_on(
            patients_src, patients_dest, sort_keys, timestamp=timestamp, write_mode=write_mode)

    if sort_assessments:
        sort_keys = ('patient_id', 'created_at')
        persistence.sort_on(
            assessments_src, assessments_dest, sort_keys, timestamp=timestamp)

        print("checking sort order")
        t0 = time.time()
        raw_patient_ids = persistence.NewFixedStringReader(assessments_dest['patient_id'])[:]
        raw_created_ats = persistence.NewTimestampReader(assessments_dest['created_at'])[:]
        last_pid = raw_patient_ids[0]
        last_cat = raw_created_ats[0]
        duplicates = 0
        for i_r in range(1, len(raw_patient_ids)):
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

    if sort_tests:
        sort_keys = ('patient_id', 'created_at')
        persistence.sort_on(
            tests_src, tests_dest, sort_keys, timestamp=timestamp)



    # Processing
    # ##########

    # Patient processing
    # ==================

    if year_from_age:
        log("year of birth -> age; 18 to 90 filter")
        t0 = time.time()
        age = persistence.NewNumericWriter(patients_dest, chunksize, 'age', timestamp, 'uint32',
                                           write_mode)
        age_filter = persistence.NewNumericWriter(patients_dest, chunksize, 'age_filter',
                                                  timestamp, 'bool', write_mode)
        age_16_to_90 = persistence.NewNumericWriter(patients_dest, chunksize, '16_to_90_years',
                                                timestamp, 'bool', write_mode)
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
                                                     timestamp, 'float32', write_mode)
        weights_filter = persistence.NewNumericWriter(patients_dest, chunksize, '40_to_200_kg',
                                                      timestamp, 'bool', write_mode)
        heights_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'height_cm_clean',
                                                     timestamp, 'float32', write_mode)
        heights_filter = persistence.NewNumericWriter(patients_dest, chunksize, '110_to_220_cm',
                                                      timestamp, 'bool', write_mode)
        bmis_clean = persistence.NewNumericWriter(patients_dest, chunksize, 'bmi_clean',
                                                  timestamp, 'float32', write_mode)
        bmis_filter = persistence.NewNumericWriter(patients_dest, chunksize, '15_to_55_bmi',
                                                   timestamp, 'bool', write_mode)

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
    sorted_assessments_src = assessments_dest if sort_assessments else assessments_src


    # if full_assessment_valid:


    if clean_temperatures:
        print("clean temperatures")
        t0 = time.time()
        temps = persistence.NewNumericReader(sorted_assessments_src['temperature'])
        temp_units = persistence.NewFixedStringReader(sorted_assessments_src['temperature_unit'])
        temps_valid = persistence.NewNumericReader(sorted_assessments_src['temperature_valid'])
        dest_temps = temps.getwriter(assessments_dest, 'temperature_c_clean', timestamp,
                                     write_mode)
        dest_temps_valid =\
            temps_valid.getwriter(assessments_dest, 'temperature_35_to_42_inclusive', timestamp,
                                  write_mode)
        dest_temps_modified =\
            temps_valid.getwriter(assessments_dest, 'temperature_modified', timestamp, write_mode)
        validate_temperature_1(35.0, 42.0,
                               temps, temp_units, temps_valid,
                               dest_temps, dest_temps_valid, dest_temps_modified)
        print(f"temperature cleaning done in {time.time() - t0}")


    if check_symptoms:
        print('check inconsistent health_status')
        t0 = time.time()
        check_inconsistent_symptoms_1(sorted_assessments_src, assessments_dest, timestamp)
        print(time.time() - t0)


    # Test processing
    # ===============


    # Daily assessments
    # =================

    if create_daily:
        print("generate daily assessments")
        t0 = time.time()
        patient_ids = persistence.get_reader_from_field(sorted_assessments_src['patient_id'])
        raw_patient_ids = patient_ids[:]
        distinct_pids = persistence.distinct(raw_patient_ids)
        print(f"{len(distinct_pids)} patients identified from assessments in {time.time() - t0}s")

        print("checking distinct assessments days per patient")
        t0 = time.time()
        distinct_days = persistence.distinct(
            persistence.get_reader_from_field(sorted_assessments_src['created_at_day'])[:])
        print(f"{len(distinct_days)} dates generated in {time.time() - t0}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='the dataset to load')
    parser.add_argument('-d', '--destination', help='the dataset to write results to')
    parser.add_argument('--sort', default=False, action='store_true')
    parser.add_argument('--process', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')
    args = parser.parse_args()

    data_schema = data_schemas.DataSchema(1)
    parsing_schema = parsing_schemas.ParsingSchema(1)
    timestamp = str(datetime.now(timezone.utc))

    if args.sort + args.process + args.all > 1:
        raise ValueError("At most one of '--sort', '--daily', and '--all' may be set")
    elif args.sort + args.process + args.all == 0:
        flags = 'all'
    else:
        if args.sort is True:
            flags = 'sort'
        elif args.process is True:
            flags = 'process'
        elif args.all is True:
            flags = 'all'

    with h5py.File(args.source, 'r') as ds:
        with h5py.File(args.destination, 'w') as ts:
            postprocess(ds, ts, data_schema, parsing_schema, timestamp, flags=flags)
