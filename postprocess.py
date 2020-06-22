from datetime import datetime, timezone
import time
import math

import numpy as np
import h5py
from numba import jit, njit

import utils
from processing.age_from_year_of_birth import calculate_age_from_year_of_birth_fast
from processing.weight_height_bmi import weight_height_bmi_fast_1
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

    year_from_age = True
    weight_height_bmi = True
    # post process patients
    # TODO: need an transaction table

    print(patients_src.keys())

    print(patients_src['weight_kg'].attrs['chunksize'])

    yobs = patients_src['year_of_birth']
    syobs = list()
    for y in persistence.IndexedStringReader(yobs, converter=lambda x: x.tobytes().decode()):
        syobs.append(y)
    print('len yobs:', len(syobs))
    t0 = time.time()
    byobs = np.frombuffer(''.join(syobs).encode(), dtype="S1")
    print("byobs:", time.time() - t0)
    t0 = time.time()

    print(len(byobs))
    print(byobs[0:50])


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
        age = persistence.NumericWriter2(patients_dest, chunksize, 'age', timestamp, 'uint32')
        age_filter = persistence.NumericWriter2(patients_dest, chunksize, '16_to_90_years',
                                                timestamp, 'bool')
        calculate_age_from_year_of_birth_fast(
            16, 90,
            patients_src['year_of_birth'],
            age, age_filter,
            2020)
        log(f"completed in {time.time() - t0}")

    if weight_height_bmi:
        log("height / weight / bmi; standard range filters")
        t0 = time.time()

        weights_clean = persistence.NumericWriter2(patients_dest, chunksize, 'weight_kg_clean',
                                                   timestamp, 'float32')
        weights_filter = persistence.NumericWriter2(patients_dest, chunksize, '40_to_200_kg',
                                                    timestamp, 'bool')
        heights_clean = persistence.NumericWriter2(patients_dest, chunksize, 'height_cm_clean',
                                                   timestamp, 'float32')
        heights_filter = persistence.NumericWriter2(patients_dest, chunksize, '110_to_220_cm',
                                                    timestamp, 'bool')
        bmis_clean = persistence.NumericWriter2(patients_dest, chunksize, 'bmi_clean',
                                                timestamp, 'float32')
        bmis_filter = persistence.NumericWriter2(patients_dest, chunksize, '15_to_55_bmi',
                                                    timestamp, 'bool')

        weight_height_bmi_fast_1(40, 200, 110, 220, 15, 55,
                            None, None,
                            # persistence.NumericReader(patients_src['weight_kg']),
                            # persistence.NumericReader(patients_src['height_cm']),
                            # persistence.NumericReader(patients_src['bmi']),
                            patients_src['weight_kg'],
                            patients_src['height_cm'],
                            patients_src['bmi'],
                            weights_clean, weights_filter, None,
                            heights_clean, heights_filter, None,
                            bmis_clean, bmis_filter, None)
        log(f"completed in {time.time() - t0}")



    print('check health_status')


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
