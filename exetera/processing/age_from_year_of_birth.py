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

from exetera.core.utils import check_input_lengths
from exetera.core import persistence as persist


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


def calculate_age_from_year_of_birth(destination,
                                     year_of_birth, in_range_fn, current_year,
                                     chunksize=None, timestamp=None, name='age'):
    age_18_to_90 = persist.NumericWriter(
        destination, chunksize, '18_to_90', timestamp, 'uint8',
        needs_filter=True)
    age = persist.NumericWriter(
        destination, chunksize, 'age', timestamp, 'int32')
    for y in persist.IndexedStringReader(year_of_birth):
        try:
            a = current_year - int(float(y))
            age_18_to_90.append(in_range_fn(a))

        except ValueError:
            age_18_to_90.append(0)
            age.append(None)


def calculate_age_from_year_of_birth_fast(datastore, min_age, max_age,
                                          year_of_birth, year_of_birth_filter,
                                          age, age_filter, age_range_filter, year,
                                          chunksize=None, timestamp=None):
    yob_v = datastore.get_reader(year_of_birth)
    yob_f = datastore.get_reader(year_of_birth_filter)
    raw_ages = year - yob_v[:]
    raw_age_filter = yob_f[:]
    raw_age_range_filter = raw_age_filter & (min_age <= raw_ages) & (raw_ages <= max_age)
    age.write_part(raw_ages)
    age_filter.write_part(raw_age_filter)
    age_range_filter.write_part(raw_age_range_filter)
    age.flush()
    age_filter.flush()
    age_range_filter.flush()


    # length = len(yob_v)
    # chunksize = age.chunksize if chunksize is None else chunksize
    # for c in range(math.ceil(len(yob_v)) / chunksize):
    #     src_index_start = c * chunksize
    #     maxindex =\
    #         chunksize if (c+1) * chunksize <= length else length % chunksize
    #     src_index_end = src_index_start + maxindex
    #
    #     src_yob_vals = yob_v[src_index_start:src_index_end]
    #     src_yob_flts = yob_f[src_index_start:src_index_end]
    #     raw_ages = src_yob_vals - year
    #     age.values[:maxindex] = raw_ages
    #     age_filter.values[:maxindex] =\
    #         np.logical_not(src_yob_flts) & (min_age <= raw_ages) & (raw_ages <= max_age)
    #     age.write_chunk(maxindex)
    #     age_filter.write_chunk(maxindex)
    #
    # age.flush()
    # age_filter.flush()
