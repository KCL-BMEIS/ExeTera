from collections import defaultdict

import h5py
import itertools
import scipy

from exetera.core.session import Session
from exetera.core import utils

class Result:
    def __init__(self):
        self.count = 0
        self.positive = 0
        self.negative = 0
        self.predicted_positive = 0
        self.predicted_negative = 0

    def update(self, old_tr, new_tr, prediction):
        self.count += 1
        if old_tr == 3 or new_tr == 4:
            self.positive += 1
        elif old_tr == 2 or new_tr == 3:
            self.negative += 1

        if prediction == 0:
            self.predicted_negative += 1
        else:
            self.predicted_positive += 1

    def __str__(self):
        return "count {}; neg {}; pos {}; -pred {}; +pred {}".format(
            self.count, self.negative, self.positive,
            self.predicted_negative, self.predicted_positive)


def supplement_paper_processing_2(ds, src_data):
    print(src_data['filtered_patients'].keys())
    print(src_data['final_assessments'].keys())
    ptnts = src_data['filtered_patients']
    asmts = src_data['final_assessments']
    supplement_fields = ('vs_multivitamins', 'vs_none', 'vs_omega_3',
                         'vs_vitamin_c', 'vs_vitamin_d', 'vs_zinc',)

    vs_none = ds.get(ptnts['vs_none']).data[:]
    print(utils.build_histogram(vs_none))
    vs_omega_3 = ds.get(ptnts['vs_omega_3']).data[:]
    print(utils.build_histogram(vs_omega_3))
    vs_vitamin_c = ds.get(ptnts['vs_vitamin_c']).data[:]
    print(utils.build_histogram(vs_vitamin_c))
    vs_vitamin_d = ds.get(ptnts['vs_vitamin_d']).data[:]
    print(utils.build_histogram(vs_vitamin_d))
    vs_zinc = ds.get(ptnts['vs_zinc']).data[:]
    print(utils.build_histogram(vs_zinc))
    vs_multivitamins = ds.get(ptnts['vs_multivitamins']).data[:]
    print(utils.build_histogram(vs_multivitamins))

    old_result = ds.get(ptnts['old_test_result']).data[:]
    print(utils.build_histogram(old_result))
    new_result = ds.get(ptnts['new_test_result']).data[:]
    print(utils.build_histogram(new_result))
    prediction = ds.get(asmts['prediction']).data[:]
    print(utils.build_histogram(prediction))

    vit_categories = defaultdict(Result)
    for i in range(len(vs_none)):
        key = (vs_none[i], vs_multivitamins[i], vs_omega_3[i],
               vs_vitamin_c[i], vs_vitamin_d[i], vs_zinc[i])
        vit_categories[key].update(old_result[i], new_result[i], prediction[i])

    results = sorted(list(vit_categories.items()))

    print("(n, m, o, c, d, z)")
    for k, v in results:
        print(k, v)
        print(k, v.positive / (v.negative + v.positive))

    all_positive = (old_result == 3) | (new_result == 4)
    all_negative = (all_positive == False) & ((old_result == 2) | (new_result == 3))
    all_positive_ratio = all_positive.sum() / (all_positive.sum() + all_negative.sum())
    print("overall tested:", all_positive_ratio)
    all_ppositive_ratio = prediction.sum() / len(prediction)
    print("overall predicted:", all_ppositive_ratio)
    neg = 0
    pos = 0
    pneg = 0
    ppos = 0
    for k, v in results:
        if k[0] == 1:
            neg += v.negative
            pos += v.positive
            pneg += v.predicted_negative
            ppos += v.predicted_positive
    pos_ratio = pos / (neg + pos)
    print('v_none tested:', pos_ratio, pos_ratio / all_positive_ratio)
    ppos_ratio = ppos / (pneg + ppos)
    print('v_none predicted:', ppos_ratio, ppos_ratio / all_ppositive_ratio)

    neg = 0
    pos = 0
    pneg = 0
    ppos = 0
    for k, v in results:
        if k[1] == 1:
            neg += v.negative
            pos += v.positive
            pneg += v.predicted_negative
            ppos += v.predicted_positive
    pos_ratio = pos / (neg + pos)
    print('v_multi tested:', pos_ratio, pos_ratio / all_positive_ratio)
    print('v_multi predicted:', ppos_ratio, ppos_ratio / all_ppositive_ratio)

    neg = 0
    pos = 0
    pneg = 0
    ppos = 0
    for k, v in results:
        if k[1] == 1 or k[3] == 1:
            neg += v.negative
            pos += v.positive
            pneg += v.predicted_negative
            ppos += v.predicted_positive
    pos_ratio = pos / (neg + pos)
    print('v_multi | v_vitamin_c tested:', pos_ratio, pos_ratio / all_positive_ratio)
    ppos_ratio = ppos / (pneg + ppos)
    print('v_multi | v_vitamin_c predicted:', ppos_ratio, ppos_ratio / all_ppositive_ratio)

    neg = 0
    pos = 0
    pneg = 0
    ppos = 0
    for k, v in results:
        if k[1] == 1 or k[4] == 1:
            neg += v.negative
            pos += v.positive
            pneg += v.predicted_negative
            ppos += v.predicted_positive
    pos_ratio = pos / (neg + pos)
    print('v_multi | v_vitamin_d tested:', pos_ratio, pos_ratio / all_positive_ratio)
    ppos_ratio = ppos / (pneg + ppos)
    print('v_multi | v_vitamin_d predicted:', ppos_ratio, ppos_ratio / all_ppositive_ratio)




if __name__ == '__main__':
    datastore = Session()
    src_file = '/home/ben/covid/ds_20200830_supplements.hdf5'
    with h5py.File(src_file, 'r') as src_data:
        supplement_paper_processing_2(datastore, src_data)
