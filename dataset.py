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

import numpy_buffer


class Dataset:

    def __init__(self, source, field_descriptors=None, progress=False, stop_after=None):
        self.names_ = list()
        self.fields_ = list()
        self.names_ = list()
        self.index_ = None

        if source:
            csvf = csv.DictReader(source, delimiter=',', quotechar='"')
            self.names_ = csvf.fieldnames

        transforms_by_index = list()
        new_fields = list()
        for i_n, n in enumerate(self.names_):
            if field_descriptors and n in field_descriptors:
                transforms_by_index.append(field_descriptors[n])
                to_datatype = field_descriptors[n].to_datatype
                if to_datatype == str:
                    new_fields.append(list())
                else:
                    new_fields.append(numpy_buffer.NumpyBuffer(dtype=to_datatype))
            else:
                transforms_by_index.append(None)
                new_fields.append(list())

        # self.new_fields = [None] * len(self.names_)
        # for i_t, t in enumerate(transforms_by_index):
        #     self.new_fields[i_t] = [None] *

        # read the cvs rows into the fields
        csvf = csv.reader(source, delimiter=',', quotechar='"')
        ecsvf = iter(csvf)
        for i, fields in enumerate(ecsvf):
            for i_f, f in enumerate(fields):
                t = transforms_by_index[i_f]
                new_fields[i_f].append(f if not t else t.strings_to_values[f])
            del fields
            if progress:
                if i % 100000 == 0:
                    print(i)
            if stop_after and i >= stop_after:
                break
        if progress:
            print(i)

        # assign the built sequences to fields_
        for i_f, f in enumerate(new_fields):
            if isinstance(f, list):
                self.fields_.append(f)
            else:
                self.fields_.append(f.finalise())
        self.index_ = np.asarray([i for i in range(len(self.fields_[0]))], dtype=np.uint32)

        #     if i > 0 and i % lines_per_dot == 0:
        #         if i % (lines_per_dot * newline_at) == 0:
        #             print(f'. {i}')
        #         else:
        #             print('.', end='')
        # if i % (lines_per_dot * newline_at) != 0:
        #     print(f' {i}')

    def sort(self, keys):
        #map names to indices
        kindices = [self.field_to_index(k) for k in keys]

        def index_sort(indices):
            def inner_(r):
                t = tuple(self.fields_[i][r] for i in indices)
                return t
            return inner_

        self.index_ = sorted(self.index_, key=index_sort(kindices))

        for i_f in range(len(self.fields_)):
            unsorted_field = self.fields_[i_f]
            self.fields_[i_f] = Dataset._apply_permutation(self.index_, unsorted_field)
            del unsorted_field

    @staticmethod
    def _apply_permutation(permutation, field):
        # n = len(permutation)
        # for i in range(0, n):
        #     print(i)
        #     pi = permutation[i]
        #     while pi < i:
        #         pi = permutation[pi]
        #     fields[i], fields[pi] = fields[pi], fields[i]
        # return fields
        if isinstance(field, list):
            sorted_field = [None] * len(field)
            for ip, p in enumerate(permutation):
                sorted_field[ip] = field[p]
        else:
            sorted_field = np.empty_like(field)
            for ip, p in enumerate(permutation):
                sorted_field[ip] = field[p]
        return sorted_field

    def field_by_name(self, field_name):
        return self.fields_[self.field_to_index(field_name)]

    def field_to_index(self, field_name):
        return self.names_.index(field_name)

    def value(self, row_index, field_index):
        return self.fields_[field_index][row_index]

    def value_from_fieldname(self, index, field_name):
        return self.fields_[self.field_to_index(field_name)][index]

    def row_count(self):
        return len(self.index_)

    def show(self):
        for ir, r in enumerate(self.names_):
            print(f'{ir}-{r}')
