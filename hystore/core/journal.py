import numpy as np
import h5py

from hystore.core import operations as ops
from hystore.core import fields as flds


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

    old_ids = session.get(old_src[src_pk]).data[:]
    old_ids_valid_to = session.get(old_src['j_valid_to']).data[:]
    old_sorted_index = session.dataset_sort_index((old_ids, old_ids_valid_to))

    new_ids = session.get(new_src[src_pk]).data[:]
    new_sorted_index = session.dataset_sort_index((new_ids,))

    # get the row maps for rows that we need to compare
    old_map, new_map = ops.ordered_generate_journalling_indices(old_ids, new_ids)
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
            # old_f_i_ = old_f.indices[:]
            # old_f_v_ = old_f.values[:]
            new_f_i_, new_f_v_ = session.apply_index(new_sorted_index, new_f)
            # new_f_i_ = new_f.indices[:]
            # new_f_v_ = new_f.values[:]
            ops.compare_indexed_rows_for_journalling(old_map, new_map,
                                                     old_f_i_, old_f_v_, new_f_i_, new_f_v_,
                                                     to_keep)
        else:
            old_f_ = session.apply_index(old_sorted_index, old_f)
            # old_f_ = old_f.data[:]
            new_f_ = session.apply_index(new_sorted_index, new_f)
            # new_f_ = new_f.data[:]
            ops.compare_rows_for_journalling(old_map, new_map, old_f_, new_f_, to_keep)

        print("to_keep:", to_keep.astype(np.uint8))
        print(to_keep.sum(), len(to_keep))

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
                for t in tables:
                    print("journaling {}".format(t))
                    journal_table(session, schema[t], old_src[t], new_src[t],
                                  'id', dest.create_group(t))
