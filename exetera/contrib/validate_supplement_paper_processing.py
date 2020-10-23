from collections import defaultdict

import numpy as np


from exetera.core import dataset, utils

# keys = ('health_worker_with_contact', 'asmt_prediction', 'old_test_result', 'new_test_result')

meanings = {
    ('0', '3'): 1,
    ('0', '4'): 2,
    ('1', '3'): 1,
    ('1', '4'): 2,
    ('2', '0'): 1,
    ('2', '1'): 1,
    ('2', '2'): 1,
    ('2', '3'): 1,
    ('2', '4'): 2,
    ('3', '0'): 2,
    ('3', '1'): 2,
    ('3', '2'): 2,
    ('3', '3'): 2,
    ('3', '4'): 2
}

#filename = '/home/ben/OneDrive/supplement_paper/supplements_patients_20200731_1.csv'
filename = '/home/ben/covid/supplements_patients.csv'
with open(filename) as f:
    # ds = dataset.Dataset(f, keys=keys, show_progress_every=1000000)
    ds = dataset.Dataset(f, show_progress_every=250000)
    print(sorted(utils.build_histogram(ds.field_by_name('asmt_prediction'))))
    print(sorted(utils.build_histogram(ds.field_by_name('new_test_result'))))
    print(sorted(utils.build_histogram(ds.field_by_name('old_test_result'))))

    print(sorted(utils.build_histogram(ds.field_by_name('health_worker_with_contact'))))

    categories = defaultdict(int)

    otr = ds.field_by_name('old_test_result')
    ntr = ds.field_by_name('new_test_result')


    for i_r in range(len(otr)):
        categories[(otr[i_r], ntr[i_r])] += 1

    results = sorted(list(categories.items()))
    total_negative = 0
    total_positive = 0
    for r in results:
        print(r)
        if r[0] in meanings:
            if meanings[r[0]] == 1:
                total_negative += r[1]
            else:
                total_positive += r[1]

    print(total_negative)
    print(total_positive)

    asmt_keys = [n for n in ds.names_ if "asmt" in n]
    asmt_keys = [n for n in asmt_keys if n != 'asmt_prediction_score']
    for k in asmt_keys:
        f = ds.field_by_name(k)
        print(k, utils.build_histogram(sorted(f)))
