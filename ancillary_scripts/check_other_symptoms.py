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

import csv
from collections import defaultdict
import string

import dataset
import utils


def write_to_csv(output_file, csv_tuples):
    with open(output_file, 'w') as f:
        csvw = csv.writer(f)
        elem_count = len(csv_tuples[0])
        words_in_line = [None] * (elem_count + 1)
        for c in csv_tuples:
            for i_e, e in enumerate(c[0]):
                words_in_line[i_e] = e
            words_in_line[-1] = str(c[1])
            csvw.writerow(words_in_line)


class Locations:
    def __init__(self):
        self.hosp = 0
        self.bfhosp = 0

    def add(self, location):
        if location == 'hospital':
            self.hosp += 1
        elif location == 'back_from_hospital':
            self.bfhosp += 1


def check_other_symptoms(input_filename):
    with open(input_filename) as f:
        ds = dataset.Dataset(f,
                             keys=('patient_id', 'updated_at', 'other_symptoms', 'treatment',
                                   'location'),
                             show_progress_every=500000)
                             # show_progress_every=500000, stop_after=2999999)

    by_patient = defaultdict(Locations)
    p_id = ds.field_by_name('patient_id')
    other = ds.field_by_name('other_symptoms')
    treatment = ds.field_by_name('treatment')
    location = ds.field_by_name('location')

    word_dict = defaultdict(int)
    _2ple_dict = defaultdict(int)
    _3ple_dict = defaultdict(int)
    other_symptoms_empty = 0
    other_treatment_empty = 0
    table = str.maketrans('', '', string.punctuation)
    for i_r in range(len(other)):
        by_patient[p_id[i_r]].add(location[i_r])
        if other[i_r] == '':
            other_symptoms_empty += 1
        else:
            # split and clean words, then add to dictionary
            words = other[i_r].split()
            cwords = [(w.lower()).translate(table) for w in words]
            cwords = [w for w in cwords if w != '']
            for c in cwords:
                word_dict[c] += 1
            for i_c in range(len(cwords) - 1):
                _2ple_dict[(cwords[i_c], cwords[i_c + 1])] += 1
            for i_c in range(len(cwords) - 2):
                _3ple_dict[(cwords[i_c], cwords[i_c + 1], cwords[i_c + 2])] += 1
    for i_r in range(len(treatment)):
        if treatment[i_r] == '':
            other_treatment_empty += 1

    by_patient_values = [x for x in by_patient.values()]
    by_patient_hist = utils.build_histogram([(v.hosp, v.bfhosp) for v in by_patient_values])
    by_patient_hist = sorted(by_patient_hist, reverse=True)
    print(by_patient_hist)

    print(utils.build_histogram(location))
    print('other_symptoms - non-empty', ds.row_count() - other_symptoms_empty)
    print('other_treatment - non-empty', ds.row_count() - other_treatment_empty)
    by_max_freq = sorted([w for w in word_dict.items()], key=lambda x: (-x[1], x[0]))
    by_max_freq_2ple =\
        sorted([w for w in _2ple_dict.items()], key=lambda x: (-x[1], x[0]))
    by_max_freq_3ple =\
        sorted([w for w in _3ple_dict.items()], key=lambda x: (-x[1], x[0]))

    threshold = 100
    for w in by_max_freq:
        if w[1] < threshold:
            break
        print(w[0], w[1])

    for w in by_max_freq_2ple:
        if w[1] < threshold:
            break
        print(w[0], w[1])

    for w in by_max_freq_3ple:
        if w[1] < threshold:
            break
        print(w[0], w[1])


    for i_r in range(len(treatment)):
        if ',' in treatment[i_r]:
            print(i_r, treatment[i_r])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--assessment_data',
                        help='the location and name of the assessment data csv file')
    # parser.add_argument(
    #     '-ao', '--assessment_data_output',
    #     help='the location and name root of the output assessment data csv files')

    args = parser.parse_args()

    try:
        check_other_symptoms(args.assessment_data)
    except Exception as e:
        print(e)
        exit(-1)
