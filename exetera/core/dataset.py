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
import time
import numpy as np

from exetera.processing import numpy_buffer


class Dataset:
    """
    field_descriptors: a dictionary of field names to field descriptors that describe how the field
                       should be transformed when loading
    keys: a list of field names that represent the fields you wish to load and in what order they
          should be put. Leaving this blankloads all of the keys in csv column order
    """
    def __init__(self, source, field_descriptors=None, keys=None, filter_fn=None,
                 show_progress_every=False, start_from=None, stop_after=None, early_filter=None):
        self.names_ = list()
        self.fields_ = list()
        self.names_ = list()
        self.index_ = None

        csvf = csv.DictReader(source, delimiter=',', quotechar='"')
        available_keys = csvf.fieldnames

        if not keys:
            fields_to_use = available_keys
            index_map = [i for i in range(len(fields_to_use))]
        else:
            fields_to_use = keys
            index_map = [available_keys.index(k) for k in keys]

        early_key_index = None
        if early_filter is not None:
            if early_filter[0] not in available_keys:
                raise ValueError(
                    f"'early_filter': tuple element zero must be a key that is in the dataset")
            early_key_index = available_keys.index(early_filter[0])

        tstart = time.time()
        transforms_by_index = list()
        new_fields = list()

        # build a full list of transforms by index whether they are are being filtered by 'keys' or not
        for i_n, n in enumerate(available_keys):
            if field_descriptors and n in field_descriptors and\
               field_descriptors[n].strings_to_values and\
               field_descriptors[n].out_of_range_label is None:
                # transforms by csv field index
                transforms_by_index.append(field_descriptors[n])
            else:
                transforms_by_index.append(None)

        # build a new list of collections for every field that is to be loaded
        for i_n in index_map:
            if transforms_by_index[i_n] is not None:
                to_datatype = transforms_by_index[i_n].to_datatype
                if to_datatype == str:
                    new_fields.append(list())
                else:
                    new_fields.append(numpy_buffer.NumpyBuffer2(dtype=to_datatype))
            else:
                new_fields.append(list())

        # read the cvs rows into the fields
        csvf = csv.reader(source, delimiter=',', quotechar='"')
        ecsvf = iter(csvf)
        filtered_count = 0
        for i_r, row in enumerate(ecsvf):
            if show_progress_every:
                if i_r % show_progress_every == 0:
                    if filtered_count == i_r:
                        print(i_r)
                    else:
                        print(f"{i_r} ({filtered_count})")

            if start_from is not None and i_r < start_from:
                del row
                continue

            # TODO: decide whether True means filter or not filter consistently
            if early_filter is not None:
                if not early_filter[1](row[early_key_index]):
                    continue

            # TODO: decide whether True means filter or not filter consistently
            if not filter_fn or filter_fn(i_r):
                # for i_f, f in enumerate(fields):
                for i_df, i_f in enumerate(index_map):
                    f = row[i_f]
                    t = transforms_by_index[i_f]
                    try:
                        new_fields[i_df].append(f if not t else t.strings_to_values[f])
                    except Exception as e:
                        msg = "{}: key error for value {} (permitted values are {}"
                        print(msg.format(fields_to_use[i_f], f, t.strings_to_values))
                del row
                filtered_count += 1
                if stop_after and i_r >= stop_after:
                    break

        if show_progress_every:
            print(f"{i_r} ({filtered_count})")

        # assign the built sequences to fields_
        for i_f, f in enumerate(new_fields):
            if isinstance(f, list):
                self.fields_.append(f)
            else:
                self.fields_.append(f.finalise())
        self.index_ = np.asarray([i for i in range(len(self.fields_[0]))], dtype=np.uint32)
        self.names_ = fields_to_use
        print('loading took', time.time() - tstart, "seconds")

        #     if i > 0 and i % lines_per_dot == 0:
        #         if i % (lines_per_dot * newline_at) == 0:
        #             print(f'. {i}')
        #         else:
        #             print('.', end='')
        # if i % (lines_per_dot * newline_at) != 0:
        #     print(f' {i}')

    def sort(self, keys):
        #map names to indices
        if isinstance(keys, str):

            def single_index_sort(index):
                field = self.fields_[index]

                def inner_(r):
                    return field[r]

                return inner_
            self.index_ = sorted(self.index_,
                                 key=single_index_sort(self.field_to_index(keys)))
        else:

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
