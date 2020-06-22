from datetime import datetime, timezone
import time
import math

import numpy as np
import h5py
from numba import jit, njit, prange

import utils
from processing.age_from_year_of_birth import calculate_age_from_year_of_birth
from processing.weight_height_bmi import weight_height_bmi_fast_1
import data_schemas
import parsing_schemas
import persistence

def check_general_performance(dataset, destination, data_schema, process_schema, timestamp=None):

    chunksize = 1 << 20

    patients_src = dataset['patients']
    patients_dest = destination.create_group('patients')

    @njit(parallel=True)
    def total_up(arr):
        total = 0
        for i in prange(len(arr)):
            total += arr[i]
        return total

    t0 = time.time()
    total = 0
    none_count = 0
    for v in persistence.NumericReader(patients_src['weight_kg']):
        if v is None:
            none_count += 1
        else:
            total += v
    print(none_count, total, time.time() - t0)

    t0 = time.time()
    srcvals = patients_src['weight_kg']['values']
    srcfltr = patients_src['weight_kg']['values']
    dest = persistence.NumericWriter3(patients_dest, chunksize, 'foo', timestamp,
                                      'float32', needs_filter=True)
    for c in range(math.ceil(len(srcvals) / chunksize)):
        srcindexstart = c * chunksize
        maxindex = chunksize if (c+1)*chunksize < len(srcvals) else len(srcvals) % chunksize
        srcindexend = min((c+1) * chunksize, len(srcvals))
        cur_vals = srcvals[srcindexstart:srcindexend]
        cur_flt = srcfltr[srcindexstart:srcindexend]
        dest.write_part(cur_vals, cur_flt)
    dest.flush()
    print('writer 3:', time.time() - t0)

    t0 = time.time()
    total = 0
    none_count = 0
    nr = persistence.NumericReader(patients_src['weight_kg'])
    for i in range(len(nr)):
        v = nr[i]
        if v is None:
            none_count += 1
        else:
            total += v
    print(none_count, total, time.time() - t0)

    t0 = time.time()
    total = 0
    none_count = 0
    for v in patients_src['weight_kg']['values'][()]:
        if v is None:
            none_count += 1
        else:
            total += v
    print(none_count, total, time.time() - t0)

    t0 = time.time()
    hs = dataset['assessments']['health_status']['values'][()]
    total = total_up(hs)
    print('iterate:', total, time.time() - t0)

    t0 = time.time()
    hs = dataset['assessments']['health_status']['values'][()]
    # total = total_up(hs)
    total = np.sum(hs)
    print('iterate:', total, time.time() - t0)

    t0 = time.time()
    total = 0
    for i in range(math.ceil(len(dataset['assessments']['health_status']['values']) / chunksize)):
        hs = dataset['assessments']['health_status']['values'][i * chunksize:(i + 1) * chunksize]
        total += total_up(hs)
    print('explicit chunks:', total, time.time() - t0)

    stuff = np.random.rand(len(dataset['assessments']['health_status']['values']))
    print(stuff[()])
    t0 = time.time()
    assessments_dest = destination.create_group('assessments')
    assessments_dest.create_dataset('stuff',
                                    shape=(len(dataset['assessments']['health_status']['values']),),
                                    chunks=(chunksize,),
                                    data=stuff)
    print('writing:', time.time() - t0)

    t0 = time.time()
    writer = persistence.NumericWriter(assessments_dest, chunksize, 'stuff2', ts, 'float32', )
    for s in stuff:
        writer.append(s)
    print("writer:", time.time() - t0)


    @jit
    def write_to_values(src, dest, count=None):
        count = len(dest) if count is None else count
        for i in range(count):
            dest[i] = src[i]


    print(len(stuff))
    t0 = time.time()
    writer = persistence.NumericWriter2(assessments_dest, chunksize, 'stuff3', ts, 'float32')
    for c in range(math.ceil(len(stuff) / chunksize)):
        maxindex = chunksize if (c + 1) * chunksize <= len(stuff) else len(stuff) % chunksize
        write_to_values(stuff[c * chunksize:], writer.values, maxindex)
        writer.write_chunk(maxindex)
    print("writer2:", time.time() - t0)

    strstuff = [str(s) for s in stuff]

    t0 = time.time()
    floatstuff = np.zeros(len(stuff), dtype=np.float32)
    for i in range(len(stuff)):
        floatstuff[i] = float(strstuff[i])
    print('to float:', time.time() - t0)

    t0 = time.time()
    swriter = persistence.NumericWriter(assessments_dest, chunksize, 'stuff4', ts, 'float32',
                                        persistence.str_to_float, needs_filter=True)
    for s in strstuff:
        swriter.append(s)
    print('to float writer:', time.time() - t0)


    @njit
    def string_to_float_values(src, dest, count=None):
        count = len(dest) if count is None else count
        for i in range(count):
            dest[i] = np.fromstring(src[i])


    # back_to_floats = np.zeros(len(strstuff), dtype=np.float32)
    t0 = time.time()
    # string_to_float_values(strstuff, back_to_floats)
    back_to_floats = np.asarray(strstuff, dtype=np.float32)
    print('jit str to float:', time.time() - t0)

    t0 = time.time()
    total = 0
    hsv = dataset['patients']['weight_kg']['values'][()]
    hsf = dataset['patients']['weight_kg']['filter'][()]
    for h in persistence.filtered_iterator(hsv, hsf, 0.0):
        total += h
    print(total, time.time() - t0)

    t0 = time.time()
    total = 0
    for h in persistence.categorical_iterator(dataset['assessments']['health_status']):
        total += h
    print(total, time.time() - t0)

    t0 = time.time()
    total = 0
    hs = persistence.CategoricalReader(dataset['assessments']['health_status'])
    for h in hs:
        total += h
    print(total, time.time() - t0)


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
            check_general_performance(ds, ts, data_schema, parsing_schema, timestamp)
