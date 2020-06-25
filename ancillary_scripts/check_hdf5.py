import time
import numpy as np
import persistence

import h5py


filename = '/home/ben/covid/ds_20200610.hdf5'


def to_numpy(source, name, slice=None):
    group = source[name]
    index = group['index'][()]
    values = group['values'][()]
    print(values[:100])
    print(values.size)
    print(index[:100])
    result = [None] * (index.size - 1)
    for i in range(1, index.size):
        result[i-1] = values[index[i-1]:index[i]].tostring().decode()
        if i % 1000000 == 0:
            print(i)
    print(result[:100])
    for r in result:
        if len(r) > 0:
            print(r)
            break


def categorical_field_iterator(group, name):
    print('categorical_field_iterator')
    field = group[name]
    print(field.attrs.keys())
    chunksize = field.attrs['chunksize']

    values = field['values']
    chunkmax = int(values.size / chunksize)
    if values.size % chunksize != 0:
        chunkmax += 1
    for c in range(chunkmax):
        print(c * chunksize)
        if c == chunkmax - 1:
            length = values.size % chunksize
        else:
            length = chunksize
        vcur = values[c*chunksize:c*chunksize + length]
        for i in range(len(vcur)):
            yield vcur[i]


with h5py.File(filename, 'r') as hf:
    print(hf.keys())
    print(hf['assessments'].keys())
    print(hf['assessments']['id']['values'].size)
    print(hf['assessments']['version'].attrs['fieldtype'])
    print(hf['assessments']['version'].attrs['timestamp'])
    print(hf['assessments']['version'].attrs['chunksize'])
    print(hf['assessments']['version'].attrs['completed'])
    print(hf['assessments']['version'].attrs.keys())
    # to_numpy(hf['assessments'], 'version')

    def test_numeric_iterator():
        values_set = 0
        for p in persistence.numeric_iterator(hf['patients'], 'height_cm', invalid=None):
            if p is not None:
                values_set += 1
        print("'height' values set:", values_set)

    def test_categorical_iterator():
        total = 0
        for c in persistence.categorical_iterator(hf['assessments'], 'health_status'):
            total += c
        print('healthy:', total)

    def test_indexed_string_iterator():
        distinct = set()
        for s in persistence.indexed_string_iterator(hf['patients'], 'version'):
            distinct.add(s)
        print('version:', distinct)

    # t0 = time.time()
    # print('full copy')
    # total = 0
    # values = hf['assessments']['version']['index'][()]
    # for i, v in enumerate(values):
    #     if i % 100000 == 0:
    #         print(i)
    #     total += v
    # print(f"{time.time() - t0}: {total}")

    # t0 = time.time()
    # print('chunk copy')
    # total = 0
    # values = hf['assessments']['version']['index']
    # chunk = 0
    # chunkmax = int(values.size / 100000)
    # if values.size % 100000 != 0:
    #     chunkmax += 1
    # for c in range(chunkmax):
    #     print(c * 100000)
    #     if c == chunkmax - 1:
    #         length = values.size % 100000
    #     else:
    #         length = 100000
    #     vcur = values[c*100000:c*100000 + length]
    #     for i in range(len(vcur)):
    #         total += vcur[i]
    # print(f"{time.time() - t0}: {total}")

    def test_raw_performance():
        t0 = time.time()
        values = hf['assessments']['sore_throat']
        print(values.keys())
        print(values['keys'])
        print(sum(values['values'][()]))
        print(time.time() - t0)
        print(hf['tests']['patient_id']['values'])
        print(hf['tests']['patient_id']['values'].attrs.keys())

    # t0 = time.time()
    # print('stream')
    # values = hf['assessments']['version']['index']
    # for i, v in enumerate(values):
    #     if i % 100000 == 0:
    #         print(i)
    #     total += v
    # print(f"{time.time() - t0}: {total}")

    def test_sort():
        print(hf['assessments']['id']['values'].size)
        index = np.arange(hf['assessments']['id']['values'].size, dtype=np.uint32)


        print("sorting")
        t0 = time.time()
        # sindex =\
        #     persistence.dataset_sort(index,
        #                              (hf['assessments']['id']['values'][()],
        #                               hf['assessments']['created_at']['timestamps'][()]
        #                              ))
        sindex = \
            index = persistence.dataset_sort(
                # hf['assessments'],
                index,
                (hf['assessments']['patient_id'],
                 hf['assessments']['updated_at']))
        print(time.time() - t0)

        print('check order')
        pids = persistence.fixed_string_reader(hf['assessments']['patient_id'])
        cats = persistence.timestamps_reader(hf['assessments']['updated_at'])

        print('unsorted')
        for i in range(0, len(index)):
            if pids[i] == '00018da6407d86e2fd6c464bc2487d49':
                print(pids[i], cats[i])

        print('sorted')
        for s in range(0, len(sindex)):
            if pids[sindex[s]] == '00018da6407d86e2fd6c464bc2487d49':
                print(pids[sindex[s]], cats[sindex[s]])

        # lastval = (pids[sindex[0]], cats[sindex[0]])
        # for s in range(1, len(sindex)):
        #     curval = (pids[sindex[s]], cats[sindex[s]])
        #     if curval <= lastval:
        #         print(s, sindex[s], lastval, curval)
        #     lastval = curval



    # test_numeric_iterator()
    # test_categorical_iterator()
    # test_indexed_string_iterator()
    # test_raw_performance()
    test_sort()