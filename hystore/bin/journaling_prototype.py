import os
import sys
from collections import defaultdict
import csv
from datetime import datetime

import numpy as np

from hystore.core import dataset
from hystore.core import utils


"""
Journaling feature

* import initial dataset
* for subsequent snapshots
  * import 
"""


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: check_for_duplicates.py <directory> <pattern>")
        exit(1)

    show_progress_every = 500000

    filenames = sorted(fn for fn in os.listdir(sys.argv[1]) if sys.argv[2] in fn)
    # first = True
    # test_updated_at = dict()
    # update_count = defaultdict(int)
    # versions = defaultdict(str)
    # remaining = None
    # for fn in filenames:
    #     print(fn)
    #     with open(os.path.join(sys.argv[1], fn)) as f:
    #         if remaining is not None:
    #             if remaining == 0:
    #                 break
    #             remaining -= 1
    #         ds = dataset.Dataset(f, keys=('id', 'updated_at', 'version'),
    #                              show_progress_every=show_progress_every)
    #         ids = ds.field_by_name('id')
    #         uats = ds.field_by_name('updated_at')
    #         vsns = ds.field_by_name('version')
    #         new_count = 0
    #         updated_count = 0
    #         non_version_update = 0
    #         for i_id, id in enumerate(ids):
    #             ts = utils.timestamp_to_datetime(uats[i_id]).timestamp()
    #             cur_ts = test_updated_at.get(id, np.float64(0))
    #             if cur_ts == 0:
    #                 # new entry
    #                 new_count += 1
    #                 test_updated_at[id] = ts
    #                 update_count[id] = 1
    #                 versions[id] = vsns[i_id]
    #             elif test_updated_at[id] != ts:
    #                 # existing entry
    #                 updated_count += 1
    #                 test_updated_at[id] = ts
    #                 update_count[id] += 1
    #                 if versions[id] != vsns[i_id]:
    #                     versions[id] = vsns[i_id]
    #                 else:
    #                     non_version_update += 1
    #
    #         print(new_count, updated_count, non_version_update, sum(update_count.values()))
    # print(sorted(utils.build_histogram(versions.values())))

    class Entry:
        def __init__(self):
            self.data = None
            self.schema_index = -1
            self.update_count = 0

        def update(self, row, schema_index):
            self.data = row
            self.schema_index = schema_index
            self.update_count += 1

    class Schemas:
        def __init__(self):
            self.schema_list = list()
            self.schema_maps = {}
#            self.schema_only_in_b = {}

        def add_schema(self, schema):
            self.schema_list.append(schema)

        def get_common_keys(self, index_of_a, index_of_b):
            if index_of_a in self.schema_maps:
                if index_of_b in self.schema_maps[index_of_a]:
                    return self.schema_maps[index_of_a][index_of_b]
                else:
                    mappings = self._build_mapping(self.schema_list[index_of_a],
                                                   self.schema_list[index_of_b])
                    self.schema_maps[index_of_a][index_of_b] = mappings
                    return mappings
            else:
                mappings = self._build_mapping(self.schema_list[index_of_a],
                                               self.schema_list[index_of_b])
                self.schema_maps[index_of_a] = dict()
                self.schema_maps[index_of_a][index_of_b] = mappings
                return mappings

        def _build_mapping(self, schema_a, schema_b):
            # a_keys = set(k for k, _ in schema_a.items())#set(schema_a.keys())
            # b_keys = set(k for k, _ in schema_b.items())#set(schema_b.keys())
            a_keys = set(schema_a)
            b_keys = set(schema_b)
            common_keys = a_keys.intersection(b_keys)
            only_in_b = b_keys.difference(a_keys)
            common_indices = [(schema_a.index(k), schema_b.index(k)) for k in common_keys]
            common_indices.sort()
            b_indices = [schema_b.index(k) for k in only_in_b]
            b_indices.sort()
            return common_keys, common_indices, only_in_b, b_indices


    remaining = None
    current_rows = defaultdict(Entry)
    schemas = Schemas()
    schema_index = 0
    for fn in filenames:
        count_new_row = 0
        count_updated_version = 0
        count_updated_version_no_uat = 0
        count_updated_changes = 0
        count_updated_changes_no_uat = 0
        count_updated_new_fields = 0
        count_updated_new_fields_no_uat = 0
        # count_updated_without_updated_at_change = 0
        print(fn)
        with open(os.path.join(sys.argv[1], fn)) as f:
            field_change_counts = defaultdict(int)
            field_change_counts_no_uat = defaultdict(int)
            if remaining is not None:
                if remaining == 0:
                    break
                remaining -= 1
            csvr = csv.DictReader(f)
            schema = csvr.fieldnames
            # schema = {v: i for i, v in enumerate(schema)}
            schemas.add_schema(schema)
            csvr = csv.reader(f)
            if schema_index == 0:
                for i_r, r in enumerate(csvr):
                    if i_r > 0 and i_r % 100000 == 0:
                        print('.', end='')
                    current_rows[r[0]].update(r, schema_index)
            else:
                for i_r, r in enumerate(csvr):
                    if i_r > 0 and i_r % 100000 == 0:
                        print('.', end='')
                    id = r[0]
                    if id in current_rows:
                        entry = current_rows[id]
                        common, common_indices, in_b, in_b_indices =\
                            schemas.get_common_keys(entry.schema_index, schema_index)
                        updated_with_version_change = False
                        updated_with_other_change = False
                        for a, b in common_indices[4:]:
                            if entry.data[a] != r[b]:
                                if schema[a] == 'version':
                                    updated_with_version_change = True
                                else:
                                    updated_with_other_change = True
                                    break

                        if updated_with_version_change:
                            count_updated_version += 1
                            if entry.data[2] == r[2]:
                                count_updated_version_no_uat += 1
                            current_rows[id].update(r, schema_index)

                        if updated_with_other_change:
                            count_updated_changes += 1
                            if entry.data[2] == r[2]:
                                # count_updated_without_updated_at_change += 1
                                count_updated_changes_no_uat += 1
                                for c in common_indices:
                                    if entry.data[c[0]] != r[c[1]]:
                                        field_change_counts_no_uat[schema[c[1]]] += 1
                                current_rows[id].update(r, schema_index)
                            else:
                                for c in common_indices:
                                    if entry.data[c[0]] != r[c[1]]:
                                        field_change_counts[schema[c[1]]] += 1
                                current_rows[id].update(r, schema_index)
                        else:
                            if len(in_b) != 0:
                                count_updated_new_fields += 1
                                if entry.data[2] == r[2]:
                                    # count_updated_without_updated_at_change += 1
                                    count_updated_new_fields_no_uat += 1
                                    current_rows[id].update(r, schema_index)
                    else:
                        count_new_row += 1
                        current_rows[id].update(r, schema_index)
            print()
            print('n:', count_new_row,
                  ' v:', count_updated_version,
                  ' vnu:', count_updated_version_no_uat,
                  ' c:', count_updated_changes,
                  ' cnu:', count_updated_changes_no_uat,
                  ' f:', count_updated_new_fields,
                  ' fnu:', count_updated_new_fields_no_uat,
                  ' t:', sum(e.update_count for e in current_rows.values()))
            print(field_change_counts)
            print(field_change_counts_no_uat)
        schema_index += 1
