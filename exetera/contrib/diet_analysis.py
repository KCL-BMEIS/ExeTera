
from collections import defaultdict
import numpy as np
import h5py

from exetera.core.session import Session


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
        return "count {:7d}; neg {:6d}; pos {:6d}; -pred {:7d}; +pred {:6d}".format(
            self.count, self.negative, self.positive,
            self.predicted_negative, self.predicted_positive)

class Habits:
    def __init__(self):
        self.values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def update(self, n, o, m, c, d, z):
        self.values[0] |= 0 if n else 1
        self.values[1] |= 1 if n else 0
        self.values[2] |= 0 if o else 1
        self.values[3] |= 1 if o else 0
        self.values[4] |= 0 if m else 1
        self.values[5] |= 1 if m else 0
        self.values[6] |= 0 if c else 1
        self.values[7] |= 1 if c else 0
        self.values[8] |= 0 if d else 1
        self.values[9] |= 1 if d else 0
        self.values[10] |= 0 if z else 1
        self.values[11] |= 1 if z else 0

src_file = '/home/ben/covid/ds_20200929_full.hdf5'
dest_file = '/home/ben/covid/ds_diet_tmp.hdf5'
with h5py.File(src_file, 'r') as hf:
    with h5py.File(dest_file, 'w') as dest:
        s = Session()

        ptnts = hf['patients']
        print(hf['diet'].keys())
        diet = hf['diet']

        p_ids_ = s.get(hf['patients']['id']).data[:]
        d_pids_ = s.get(hf['diet']['patient_id']).data[:]
        d_pid_spans = s.get_spans(d_pids_)
        d_distinct_pids = s.apply_spans_first(d_pid_spans, d_pids_)
        d_pid_counts = s.apply_spans_count(d_pid_spans)
        print(np.unique(d_pid_counts, return_counts=True))
        p_diet_counts_new = s.create_numeric(dest, 'diet_counts_new', 'int32')
        dcs = s.merge_left(left_on=p_ids_, right_on=d_distinct_pids,
                           right_fields=(d_pid_counts,), right_writers=(p_diet_counts_new,))
        # res = np.unique(s.get(patients_dest['diet_counts']).data[:], return_counts=True)
        print(np.unique(p_diet_counts_new.data[:], return_counts=True))

        # ddtest = defaultdict(int)
        # for p in d_pids_:
        #     ddtest[p] += 1
        #
        # p_diet_counts_new_ = p_diet_counts_new.data[:]
        # mismatches = 0
        # for i in range(len(p_ids_)):
        #     if p_diet_counts_new_[i] != ddtest[p_ids_[i]]:
        #         mismatches += 1
        # print(mismatches)
        #
        # p_diet_counts_ = s.get(hf['patients']['diet_counts']).data[:]
        # mismatches = 0
        # for i in range(len(p_ids_)):
        #     if p_diet_counts_[i] != ddtest[p_ids_[i]]:
        #         mismatches += 1
        # print(mismatches)

        p_diet_counts_ = s.get(ptnts['diet_counts']).data[:]
        p_filter = p_diet_counts_ > 0

        # a_diet_counts = s.merge_left(left_on=d_pids_,
        #                              right_on=p_ids_,
        #                              right_fields=(p_diet_counts_,))[0]

        print('patient-based')
        print(np.unique(s.get(ptnts['vs_none']).data[:], return_counts=True))
        vs_none = np.where(s.get(ptnts['vs_none']).data[:] == 2, 1, 0)
        vs_none = s.apply_filter(p_filter, vs_none)
        print('vs_none:', np.unique(vs_none, return_counts=True))
        vs_omega_3 = np.where(s.get(ptnts['vs_omega_3']).data[:] == 2, 1, 0)
        vs_omega_3 = s.apply_filter(p_filter, vs_omega_3)
        print('vs_omega_3:', np.unique(vs_omega_3, return_counts=True))
        vs_multivitamins = np.where(s.get(ptnts['vs_multivitamins']).data[:] == 2, 1, 0)
        vs_multivitamins = s.apply_filter(p_filter, vs_multivitamins)
        print('vs_multivitamins:', np.unique(vs_multivitamins, return_counts=True))
        vs_vitamin_c = np.where(s.get(ptnts['vs_vitamin_c']).data[:] == 2, 1, 0)
        vs_vitamin_c = s.apply_filter(p_filter, vs_vitamin_c)
        print('vs_vitamin_c:', np.unique(vs_vitamin_c, return_counts=True))
        vs_vitamin_d = np.where(s.get(ptnts['vs_vitamin_d']).data[:] == 2, 1, 0)
        vs_vitamin_d = s.apply_filter(p_filter, vs_vitamin_d)
        print('vs_vitamin_d:', np.unique(vs_vitamin_d, return_counts=True))
        vs_zinc = np.where(s.get(ptnts['vs_zinc']).data[:] == 2, 1, 0)
        vs_zinc = s.apply_filter(p_filter, vs_zinc)
        print('vs_zinc:', np.unique(vs_zinc, return_counts=True))

        print('diet-based')
        print(np.unique(s.get(diet['takes_supplements']).data[:], return_counts=True))
        vsd_none = np.where(s.get(diet['takes_supplements']).data[:] == 2, 0, 1)
        print('vsd_none:', np.unique(vsd_none, return_counts=True))
        vsd_omega_3 = np.where(s.get(diet['supplements_omega3']).data[:] == 2, 1, 0)
        print('vsd_omega_3:', np.unique(vsd_omega_3, return_counts=True))
        vsd_multivitamins = np.where(s.get(diet['supplements_multivitamin']).data[:] == 2, 1, 0)
        print('vsd_multivitamins:', np.unique(vsd_multivitamins, return_counts=True))
        vsd_vitamin_c = np.where(s.get(diet['supplements_vitamin_c']).data[:] == 2, 1, 0)
        print('vsd_vitamin_c:', np.unique(vsd_vitamin_c, return_counts=True))
        vsd_vitamin_d = np.where(s.get(diet['supplements_vitamin_d']).data[:] == 2, 1, 0)
        print('vsd_vitamin_d:', np.unique(vsd_vitamin_d, return_counts=True))
        vsd_zinc = np.where(s.get(diet['supplements_zinc']).data[:] == 2, 1, 0)
        print('vsd_zinc:', np.unique(vsd_zinc, return_counts=True))

        diet_per_patient = defaultdict(Habits)
        for i in range(len(d_pids_)):
            diet_per_patient[d_pids_[i]].update(vsd_none[i], vsd_omega_3[i],
                                                  vsd_multivitamins[i], vsd_vitamin_c[i],
                                                  vsd_vitamin_d[i], vsd_zinc[i])

        p_vsd_none = np.zeros(len(diet_per_patient), np.int8)
        p_vsd_omega_3 = np.zeros(len(diet_per_patient), np.int8)
        p_vsd_multivitamins = np.zeros(len(diet_per_patient), np.int8)
        p_vsd_vitamin_c = np.zeros(len(diet_per_patient), np.int8)
        p_vsd_vitamin_d = np.zeros(len(diet_per_patient), np.int8)
        p_vsd_zinc = np.zeros(len(diet_per_patient), np.int8)
        l_diet_per_patient = sorted(list(diet_per_patient.items()))
        d_unique_pids = np.asarray([v[0] for v in l_diet_per_patient])
        print(d_unique_pids.dtype)

        # determine whether people have changed habits during diet questions
        for i in range(len(l_diet_per_patient)):
            e = l_diet_per_patient[i][1]
            p_vsd_none[i] = e.values[0] + 2 * e.values[1]
            p_vsd_omega_3[i] = e.values[2] + 2 * e.values[3]
            p_vsd_multivitamins[i] = e.values[4] + 2 * e.values[5]
            p_vsd_vitamin_c[i] = e.values[6] + 2 * e.values[7]
            p_vsd_vitamin_d[i] = e.values[8] + 2 * e.values[9]
            p_vsd_zinc[i] = e.values[10] + 2 * e.values[11]
        print(np.unique(p_vsd_none, return_counts=True))
        print(np.unique(p_vsd_omega_3, return_counts=True))
        print(np.unique(p_vsd_multivitamins, return_counts=True))
        print(np.unique(p_vsd_vitamin_c, return_counts=True))
        print(np.unique(p_vsd_vitamin_d, return_counts=True))
        print(np.unique(p_vsd_zinc, return_counts=True))

        # map to patient space
        filtered_p_ids = s.apply_filter(p_filter, p_ids_)
        print(len(filtered_p_ids), len(d_unique_pids))
        print(np.array_equal(filtered_p_ids, d_unique_pids))

        # now compare whether they are also consistent with patient supplement questions
        for i in range(len(filtered_p_ids)):
            p_vsd_none[i] |= 2 if vs_none[i] else 1
            p_vsd_omega_3[i] |= 2 if vs_omega_3[i] else 1
            p_vsd_multivitamins[i] |= 2 if vs_multivitamins[i] else 1
            p_vsd_vitamin_c[i] |= 2 if vs_vitamin_c[i] else 1
            p_vsd_vitamin_d[i] |= 2 if vs_vitamin_d[i] else 1
            p_vsd_zinc[i] |= 2 if vs_zinc[i] else 1
        print(np.unique(p_vsd_none, return_counts=True))
        print(np.unique(p_vsd_omega_3, return_counts=True))
        print(np.unique(p_vsd_multivitamins, return_counts=True))
        print(np.unique(p_vsd_vitamin_c, return_counts=True))
        print(np.unique(p_vsd_vitamin_d, return_counts=True))
        print(np.unique(p_vsd_zinc, return_counts=True))

        vit_categories = defaultdict(int)
        for i in range(len(vsd_none)):
            key = (vsd_none[i], vsd_multivitamins[i], vsd_omega_3[i],
                   vsd_vitamin_c[i], vsd_vitamin_d[i], vsd_zinc[i])
            vit_categories[key] += 1

        categories = sorted(list(vit_categories.items()))

        print("(n, m, o, c, d, z)")
        for k, v in categories:
            print(k, v)
            # print("{}, {}, {:.6g}, {:.6g}, {:.6g}".format(k, v, v.positive / (v.negative + v.positive),
            #                                               v.positive / v.count, (v.negative + v.positive) / v.count))
