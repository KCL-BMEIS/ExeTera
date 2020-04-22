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
import numpy as np


class Dataset:

    def __init__(self, source, progress=False, stop_after=None):
        self.names_ = list()
        self.fields_ = list()
        self.names_ = list()
        self.index_ = None

        if source:
            csvf = csv.DictReader(source, delimiter=',', quotechar='"')
            self.names_ = csvf.fieldnames

        newline_at = 10
        lines_per_dot = 100000
        # TODO: better for the Dataset to take a stream rather than a name - this allows us to unittest it from strings
        csvf = csv.reader(source, delimiter=',', quotechar='"')

        ecsvf = iter(csvf)
        # next(ecsvf)
        if not progress:
            for i, fields in enumerate(ecsvf):
                self.fields_.append(fields)
                if stop_after and i > stop_after:
                    break
        else:
            for i, fields in enumerate(ecsvf):
                self.fields_.append(fields)
                if i % 100000 == 0:
                    print(i)
                if stop_after and i >= stop_after:
                    break
            print(i)
        #     if i > 0 and i % lines_per_dot == 0:
        #         if i % (lines_per_dot * newline_at) == 0:
        #             print(f'. {i}')
        #         else:
        #             print('.', end='')
        # if i % (lines_per_dot * newline_at) != 0:
        #     print(f' {i}')
        self.index_ = np.asarray([i for i in range(len(self.fields_))], dtype=np.uint32)
        # return strings

    def sort(self, keys):
        #map names to indices
        kindices = [self.field_to_index(k) for k in keys]

        def index_sort(indices):
            def inner_(r):
                return tuple(self.fields_[r][i] for i in indices)
            return inner_

        self.index_ = sorted(self.index_, key=index_sort(kindices))
        fields2 = self.fields_[:]
        self.fields_ = Dataset._apply_permutation(self.index_, self.fields_)

    @staticmethod
    def _apply_permutation(permutation, fields):
        # n = len(permutation)
        # for i in range(0, n):
        #     print(i)
        #     pi = permutation[i]
        #     while pi < i:
        #         pi = permutation[pi]
        #     fields[i], fields[pi] = fields[pi], fields[i]
        # return fields
        sorted_fields = [0] * len(fields)
        for ip, p in enumerate(permutation):
            sorted_fields[ip] = fields[p]
        return sorted_fields

    def field_to_index(self, field_name):
        return self.names_.index(field_name)

    def value(self, row_index, field_index):
        return self.fields_[row_index][field_index]

    def value_from_fieldname(self, index, field_name):
        return self.fields_[index][self.field_to_index(field_name)]

    def row_count(self):
        return len(self.fields_)

    def show(self):
        for ir, r in enumerate(self.names_):
            print(f'{ir}-{r}')
