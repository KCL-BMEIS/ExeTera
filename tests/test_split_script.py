# Copyright 2020 KCL-BMEIS - King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import csv

import dataset

import split

def jury_rigged_split_sanity_check():
    patient_filename = '/home/ben/covid/patients_export_geocodes_20200504030002.csv'
    assessment_filename = '/home/ben/covid/assessments_export_20200504030002.csv'
    patient_keys = ('id', 'created_at', 'year_of_birth')
    assessment_keys = ('id', 'patient_id', 'created_at', 'updated_at')
    with open(patient_filename) as pds:
        p_ds = dataset.Dataset(pds, keys=patient_keys,
                               progress=True)

    p_ds.sort(keys=('created_at', 'id'))
    print(p_ds.row_count())
    patient_ids = set()
    p_ids = p_ds.field_by_name('id')
    for pid in p_ids:
        patient_ids.add(pid)

    assessment_id_counts = defaultdict(int)
    assessment_ids = 0
    orphaned_assessment_ids = 0
    with open(assessment_filename) as ads:
        a_ds = dataset.Dataset(ads, keys=assessment_keys,
                               progress=True)
    print(a_ds.row_count())
    a_pids = a_ds.field_by_name('patient_id')
    for aid in a_pids:
        if aid in patient_ids:
            assessment_ids += 1
            assessment_id_counts[aid] += 1
        else:
            orphaned_assessment_ids += 1
    print("assessment_ids:", assessment_ids)
    print("orphaned_assessment_ids:", orphaned_assessment_ids)

    small_p_dses = list()
    for i in range(8):
        with open(patient_filename[:-4] + f"_{i:04d}" + ".csv") as spds:
            small_p_dses.append(dataset.Dataset(spds, keys=patient_keys,
                                                progress=True))
    small_a_dses = list()
    for i in range(8):
        with open(assessment_filename[:-4] + f"_{i:04d}" + ".csv") as sads:
            small_a_dses.append(dataset.Dataset(sads, keys=assessment_keys,
                                                progress=True))


    print('p_ds:', p_ds.row_count())
    for i_d in range(len(small_p_dses)):
        print(f'spds{i_d}', small_p_dses[i_d].row_count())


    p_ids = p_ds.field_by_name('id')
    # p_id_indices = dict()
    # for i_p, p in enumerate(p_ids):
    #     p_id_indices[p] = i_p
    p_c_ats = p_ds.field_by_name('created_at')
    p_yobs = p_ds.field_by_name('year_of_birth')

    accumulated = 0
    patient_id_checks = 0
    small_assessment_id_counts = defaultdict(int)
    for d in range(8):
        print('checking subset', d)
        sp_ids = small_p_dses[d].field_by_name('id')
        sp_c_ats = small_p_dses[d].field_by_name('created_at')
        sp_yobs = small_p_dses[d].field_by_name('year_of_birth')
        for i_r in range(small_p_dses[d].row_count()):
            if sp_ids[i_r] != p_ids[accumulated + i_r]:
                print(i_r, 'ids do not match')
            if sp_c_ats[i_r] != p_c_ats[accumulated + i_r]:
                print(i_r, 'updated ats do not match')
            if sp_yobs[i_r] != p_yobs[accumulated + i_r]:
                print(i_r, 'year of births do not match')
        accumulated += small_p_dses[d].row_count()

        # check all assessments in each bucket match up with patients in that bucket
        spatient_ids = set()
        for pid in sp_ids:
            spatient_ids.add(pid)
        sa_ids = small_a_dses[d].field_by_name('patient_id')
        for i_r in range(len(sa_ids)):
            pid = sa_ids[i_r]
            small_assessment_id_counts[pid] += 1
            if pid not in spatient_ids:
                print(i_r, 'patient id not in patient dataset')
            patient_id_checks += 1
    print('patient id checks:', patient_id_checks)

jury_rigged_split_sanity_check()
