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

from utils import check_input_lengths
import persistence as persist


class CalculateAgeFromYearOfBirth:

    def __init__(self, f_missing_age, f_bad_age, in_range_fn, current_year):
        self.f_missing_age = f_missing_age
        self.f_bad_age = f_bad_age
        self.in_range_fn = in_range_fn
        self.current_year = current_year

    def __call__(self, year_of_birth, age, flags):
        check_input_lengths(('year_of_birth', 'age'), (year_of_birth, age))

        for i_r in range(len(year_of_birth)):
            yob = year_of_birth[i_r]
            if yob != '':
                a = self.current_year - int(float(yob))
                if not self.in_range_fn(a):
                    flags[i_r] |= self.f_bad_age
                    age[i_r] = 0
                else:
                    age[i_r] = a
            else:
                flags[i_r] |= self.f_missing_age


def calculate_age_from_year_of_birth(patients, year_of_birth, in_range_fn, current_year,
                                     chunksize=None, timestamp=None, name='age'):
    bad_age = persist.NumericWriter(
        patients, chunksize, 'bad_age', timestamp, 'numeric,uint8')
    age = persist.NumericWriter(
        patients, chunksize, 'age', timestamp, 'numeric,int32')
    for y in persist.indexed_string_iterator(year_of_birth):
        try:
            a = current_year - int(float(y))
            bad_age.append(0)

        except ValueError:
            bad_age.append(1)
            age.append(None)
        if in_range_fn(a):
            bad_age.append(0)
            age.append(age)

def filter_by_range(dataset, source, name, in_range_fn, chunksize=None, timestamp=None):
    s_it = persist.iterator(source)
    flags = persist.NumericWriter(
        dataset, source.attrs['chunksize'], name, timestamp, 'uint8')
