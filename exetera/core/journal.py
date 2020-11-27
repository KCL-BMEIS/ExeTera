import time
import numpy as np
import h5py

from exetera.core import operations as ops
from exetera.core import fields as flds
from exetera.core import utils


# TODO:
# * add j_valid_from and j_valid_to as columns during import - DONE
# * add journalling operation
# * add journalling operation per table
# * add filter by timestamp operation to session

# journalling overview

# existing dataset a
# import and sort dataset b

# start with empty filter

# create map between rows in a and b

# for each common field:
#   perform merge style row comparison
#   common keys must be compared
#   new keys must be added
#   old keys must have their 'to' date set

# for each new field
#   field is not present in dataset until the timestamp

# for each old field
#   field has its 'to' value set everywhere


def journal_table(session, schema, old_src, new_src, src_pk, result):
    old_keys = set(old_src.keys())
    new_keys = set(new_src.keys())

    common_keys = old_keys.intersection(new_keys)
    common_keys.remove('j_valid_from')
    common_keys.remove('j_valid_to')
    old_only_keys = old_keys.difference(new_keys)
    new_only_keys = new_keys.difference(old_keys)

    with utils.Timer("sorting old ids"):
        old_ids = session.get(old_src[src_pk])
        old_ids_ = old_ids.data[:]
        old_ids_valid_from = session.get(old_src['j_valid_from']).data[:]
        old_sorted_index = session.dataset_sort_index((old_ids_, old_ids_valid_from))
    old_count = len(old_ids_)

    with utils.Timer("sorting new_ids"):
        new_ids_ = session.get(new_src[src_pk]).data[:]
        new_sorted_index = session.dataset_sort_index((new_ids_,))
    new_count = len(new_ids_)

    # print("old_ids:", old_ids_[old_sorted_index[:20]])
    # print("new_ids:", new_ids_[new_sorted_index[:20]])

    # get the row maps for rows that we need to compare
    with utils.Timer("generating row_maps for merging"):
        old_ids_ = old_ids_[old_sorted_index]
        new_ids_ = new_ids_[new_sorted_index]
        old_map, new_map = ops.ordered_generate_journalling_indices(old_ids_, new_ids_)

    to_keep = np.zeros(len(old_map), dtype=np.bool)

    schema_fields = schema.fields.keys()
    common_keys = [k for k in schema_fields if k in common_keys]
    print("old_map:", old_map)
    print("new_map:", new_map)

    for k in common_keys:
        if k in (src_pk, 'j_valid_from', 'j_valid_to'):
            continue
        old_f = session.get(old_src[k])
        new_f = session.get(new_src[k])
        print(k)
        if isinstance(old_f, flds.IndexedStringField):
            old_f_i_, old_f_v_ = session.apply_index(old_sorted_index, old_f)
            new_f_i_, new_f_v_ = session.apply_index(new_sorted_index, new_f)
            ops.compare_indexed_rows_for_journalling(old_map, new_map,
                                                     old_f_i_, old_f_v_, new_f_i_, new_f_v_,
                                                     to_keep)
        else:
            old_f_ = session.apply_index(old_sorted_index, old_f)
            new_f_ = session.apply_index(new_sorted_index, new_f)
            ops.compare_rows_for_journalling(old_map, new_map, old_f_, new_f_, to_keep)

        print("to_keep:", to_keep.astype(np.uint8))
        print(to_keep.sum(), len(to_keep))

    merged_length = len(old_ids.data) + to_keep.sum()

    only_in_old = 0
    only_in_new = 0
    not_updated = 0
    updated = 0
    for i in range(len(old_map)):
        if old_map[i] == -1:
            only_in_new += 1
        if new_map[i] == -1:
            only_in_old += 1
        if (old_map[i] != -1) and (to_keep[i] == True):
            updated += 1
        if (new_map[i] != -1) and (to_keep[i] == False):
            not_updated += 1

    for k in common_keys:
        if k in (src_pk, 'j_valid_from', 'j_valid_to'):
            continue
        old_f = session.get(old_src[k])
        new_f = session.get(new_src[k])
        print(k)
        if isinstance(old_f, flds.IndexedStringField):
            old_f_i_, old_f_v_ = session.apply_index(old_sorted_index, old_f)
            new_f_i_, new_f_v_ = session.apply_index(new_sorted_index, new_f)
            dest_i_ = np.zeros(merged_length + 1, old_f_i_.dtype)
            val_count = ops.merge_indexed_journalled_entries_count(old_map, new_map, to_keep,
                                                                   old_f_i_, new_f_i_)
            dest_v_ = np.zeros(val_count, old_f_v_.dtype)
            ops.merge_indexed_journalled_entries(old_map, new_map, to_keep,
                                                 old_f_i_, old_f_v_, new_f_i_, new_f_v_,
                                                 dest_i_, dest_v_)
            dest_f = new_f_i_.create_like(result, k)
            dest_f.indices.write(dest_i_)
            dest_f.values.write(dest_v_)

        else:
            old_f_ = session.apply_index(old_sorted_index, old_f)
            new_f_ = session.apply_index(new_sorted_index, new_f)
            dest_ = np.zeros(merged_length, old_f_.dtype)
            ops.merge_journalled_entries(old_map, new_map, to_keep, old_f_, new_f_, dest_)
            dest_f = new_f_.create_like(result, k)
            dest_f.data.write(dest_)

    print("old_count:", old_count)
    print("new_count:", new_count)
    print("only in old:", only_in_old)
    print("only in new:", only_in_new)
    print("updated:", updated)
    print("not updated:", not_updated)
    print("post journal count:", merged_length)

    # merge the tables - the new field length is the original table field length + the number
    # of elements from the new table that are being kept




def journal_test_harness(session, schema, old_file, new_file, dest_file):

    with h5py.File(old_file, 'r') as old_src:
        with h5py.File(new_file, 'r') as new_src:
            with h5py.File(dest_file, 'w') as dest:
                old_tables = set(old_src.keys())
                new_tables = set(new_src.keys())
                if old_tables != new_tables:
                    msg = "Old and new datasets must have the same tables {} and {} respectively"
                    raise ValueError(msg.format(old_tables, new_tables))

                tables = [k for k in schema.keys() if k in old_tables]
                # tables = ['diet']
                for t in tables:
                    print("journaling {}".format(t))
                    t0 = time.time()
                    journal_table(session, schema[t], old_src[t], new_src[t],
                                  'id', dest.create_group(t))
                    t1 = time.time()
                    print("test finished in {} seconds", t1 - t0)
