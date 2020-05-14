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

import dataset
import utils


filename = '/home/ben/covid/assessments_export_20200504030002.csv'

with open(filename) as f:
    a_ds = dataset.Dataset(f,
                           keys=('id', 'patient_id', 'created_at', 'had_covid_test',
                                 'tested_covid_positive'),
                           show_progress_every=500000)

class TestCount:
    def __init__(self):
        self.count = 0
        self.had_covid_test_count = 0
        self.tcp_waiting = 0
        self.tcp_no = 0
        self.tcp_yes = 0
        self.last_yes_date = None
        self.last_no_date = None
        self.latest_assessment_date = None

    def add_assessment(self, had_covid_test, tested_covid_positive, created_at):
        self.count += 1
        if had_covid_test.lower() == 'true':
            self.had_covid_test_count += 1

        if tested_covid_positive == 'waiting':
            self.tcp_waiting += 1
        elif tested_covid_positive == 'no':
            self.tcp_no += 1
            if self.last_no_date is None:
                self.last_no_date = created_at
            else:
                self.last_no_date = max(self.last_no_date, created_at)
        elif tested_covid_positive == 'yes':
            self.tcp_yes += 1
            if self.last_yes_date is None:
                self.last_yes_date = created_at
            else:
                self.last_yes_date = max(self.last_yes_date, created_at)
        if self.latest_assessment_date is None:
            self.latest_assessment_date = created_at
        else:
            self.latest_assessment_date = max(self.latest_assessment_date, created_at)


by_patient = defaultdict(TestCount)
a_pids = a_ds.field_by_name('patient_id')
a_hcts = a_ds.field_by_name('had_covid_test')
a_tcps = a_ds.field_by_name('tested_covid_positive')
a_c_ats = a_ds.field_by_name('created_at')

print(utils.build_histogram(a_hcts))
for i_r in range(a_ds.row_count()):
    by_patient[a_pids[i_r]].add_assessment(a_hcts[i_r], a_tcps[i_r], a_c_ats[i_r])

asmt_count = 0
hct_count = 0
count_yes_only = 0
count_no_only = 0
count_yes_and_no = 0
count_yes_after_no = 0
count_no_after_yes = 0
hgram_test_counts = defaultdict(int)
for k, v in by_patient.items():
    asmt_count += v.count
    if v.had_covid_test_count > 0:
        hct_count += 1
    if v.tcp_yes > 0 and v.tcp_no > 0:
        count_yes_and_no += 1
        if v.last_no_date < v.last_yes_date:
            count_yes_after_no += 1
        else:
            count_no_after_yes += 1
        # print(f"{k}: yes={v.tcp_yes}, no={v.tcp_no}")
    elif v.tcp_yes > 0:
        count_yes_only += 1
    elif v.tcp_no > 0:
        count_no_only += 1

    hgram_test_counts[v.had_covid_test_count] += 1

print("assessment record cound sanity check", asmt_count)
print("patients with 1+ 'had_covid_test' == True", hct_count)
print("patients with yes only test results", count_yes_only)
print("patients with no only test results", count_no_only)
print("patients with both yes and no test results", count_yes_and_no)
print("patients with 'yes' after 'no' results", count_yes_after_no)
print("patients with 'no' after 'yes' results", count_no_after_yes)
print(sorted([v for v in hgram_test_counts.items()], reverse=True))
