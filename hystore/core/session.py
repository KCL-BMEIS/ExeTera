import os
import uuid
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd

from hystore.core import persistence as per


# TODO:
"""
 * joins: get mapping and map multiple fields using mapping
   * can use pandas for initial functionality and then improve
   * for sorted, can use fast
   * aggregation
     * aggregate and join / join and aggregate
       * if they resolve to being the same thing, best to aggregate first
 * everything can accept groups
 * groups are registered and given names
 * indices can be built from merging pk and fks and mapping to values
 * everything can accept tuples instead of groups and operate on all of them
 * FilteredReader / FilteredWriter for applying filters without copying data
"""

class Session:

    def __init__(self, chunksize=per.DEFAULT_CHUNKSIZE,
                 timestamp=str(datetime.now(timezone.utc))):
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.chunksize = chunksize
        self.timestamp = timestamp


    def set_timestamp(self, timestamp=str(datetime.now(timezone.utc))):
        if not isinstance(timestamp, str):
            error_str = "'timestamp' must be a string but is of type {}"
            raise ValueError(error_str.format(type(timestamp)))
        self.timestamp = timestamp


    # TODO: fields is being ignored at present
    def sort_on(self, src_group, dest_group, keys,
                timestamp=datetime.now(timezone.utc), write_mode='write'):

        readers = tuple(self.get_reader(src_group[f]) for f in keys)
        t1 = time.time()
        sorted_index = self.dataset_sort_index(
            np.arange(len(readers[0]), dtype=np.uint32), readers)
        print(f'sorted {keys} index in {time.time() - t1}s')

        t0 = time.time()
        for k in src_group.keys():
            t1 = time.time()
            r = self.get_reader(src_group[k])
            w = r.get_writer(dest_group, k, timestamp, write_mode=write_mode)
            self.apply_sort_index(sorted_index, r, w)
            del r
            del w
            print(f"  '{k}' reordered in {time.time() - t1}s")
        print(f"fields reordered in {time.time() - t0}s")


    def dataset_sort_index(self, sort_indices, index=None):
        per._check_all_readers_valid_and_same_type(sort_indices)
        r_readers = tuple(reversed(sort_indices))

        raw_data = per._raw_array_from_parameter(self, 'readers', r_readers[0])

        if index is None:
            raw_index = np.arange(len(raw_data))
        else:
            raw_index = per._raw_array_from_parameter(self, 'index', index)

        acc_index = raw_index
        fdata = raw_data
        index = np.argsort(fdata, kind='stable')
        acc_index = acc_index[index]

        for r in r_readers[1:]:
            raw_data = per._raw_array_from_parameter(self, 'readers', r)
            fdata = raw_data[acc_index]
            index = np.argsort(fdata, kind='stable')
            acc_index = acc_index[index]

        return acc_index


    # # TODO: index should be able to be either a reader or an ndarray
    # def apply_sort_index(self, index, reader, writer=None):
    #     index_ = per._raw_array_from_parameter(self, 'index', index)
    #     per._check_is_reader_substitute('reader', reader)
    #     per._check_is_appropriate_writer_if_set(self, 'writer', reader, writer)
    #     reader_ = per._reader_from_group_if_required(self, 'reader', reader)
    #     if isinstance(reader_, per.IndexedStringReader):
    #         src_indices = reader_.field['index'][:]
    #         src_values = reader_.field.get('values', np.zeros(0, dtype=np.uint8))[:]
    #         indices, values = per._apply_sort_to_index_values(index_, src_indices, src_values)
    #         if len(src_indices) != len(index) + 1:
    #             raise ValueError(f"'indices' (length {len(indices)}) must be one longer than "
    #                              f"'index' (length {len(index)})")
    #         if writer:
    #             writer.write_raw(indices, values)
    #         return indices, values
    #     else:
    #         reader_ = per._raw_array_from_parameter(reader_)
    #         per._check_equal_length('index', index_, 'reader', reader_)
    #         result = per._apply_sort_to_array(index_, reader_)
    #         if writer:
    #             writer.write(result)
    #         return result


    # TODO: write filter with new readers / writers rather than deleting this
    def apply_filter(self, filter_to_apply, reader, writer=None):
        filter_to_apply_ =\
            per._raw_array_from_parameter(self, 'filter_to_apply', filter_to_apply)
        per._check_is_reader_substitute('reader', reader)

        writer_ = None
        if writer is not None:
            writer_ = per._writer_from_writer_or_group(self, 'writer', writer)
            per._check_is_appropriate_writer_if_set(self, 'writer', reader, writer_)
        reader_ = per._reader_from_group_if_required(self, 'reader', reader)
        if isinstance(reader, per.IndexedStringReader):
            src_indices = reader.field['index'][:]
            src_values = reader.field.get('values', np.zeros(0, dtype=np.uint8))[:]
            if len(src_indices) != len(filter_to_apply_) + 1:
                raise ValueError(f"'indices' (length {len(indices)}) must be one longer than "
                                 f"'filter_to_apply' (length {len(filter_to_apply_)})")

            indices, values = per._apply_filter_to_index_values(filter_to_apply_,
                                                                src_indices, src_values)
            if writer_:
                writer_.write_raw(indices, values)
            return indices, values
        else:
            reader_ = per._raw_array_from_parameter(self, 'reader', reader_)
            per._check_equal_length('filter_to_apply', filter_to_apply_, 'reader', reader_)
            result = reader_[filter_to_apply_]
            if writer_:
                writer_.write(result)
            return result


    def apply_index(self, index_to_apply, reader, writer=None):
        index_to_apply_ = per._raw_array_from_parameter(self, 'index_to_apply', index_to_apply)
        per._check_is_reader_substitute('reader', reader)
        writer_ = None
        if writer is not None:
            writer_ = per._writer_from_writer_or_group(self, 'writer', writer)
            per._check_is_appropriate_writer_if_set(self, 'writer', reader, writer_)
        reader_ = per._reader_from_group_if_required(self, 'reader', reader)
        if isinstance(reader, per.IndexedStringReader):
            src_indices = reader.field['index'][:]
            src_values = reader.field.get('values', np.zeros(0, dtype=np.uint8))[:]
            if len(src_indices) != len(index_to_apply_) + 1:
                raise ValueError(f"'indices' (length {len(indices)}) must be one longer than "
                                 f"'index_filter' (length {len(index_filter)})")

            indices, values = per._apply_indices_to_index_values(index_to_apply,
                                                                 src_indices, src_values)
            if writer_:
                writer_.write_raw(indices, values)
            return indices, values
        else:
            reader_ = per._raw_array_from_parameter(self, 'reader', reader_)
            per._check_equal_length('index_to_apply', index_to_apply_, 'reader', reader_)
            result = reader_[index_to_apply]
            if writer_:
                writer_.write(result)
            return result


    # TODO: write distinct with new readers / writers rather than deleting this
    def distinct(self, field=None, fields=None, filter=None):
        if field is None and fields is None:
            return ValueError("One of 'field' and 'fields' must be set")
        if field is not None and fields is not None:
            return ValueError("Only one of 'field' and 'fields' may be set")

        if field is not None:
            return np.unique(field)

        entries = [(f'{i}', f.dtype) for i, f in enumerate(fields)]
        unified = np.empty_like(fields[0], dtype=np.dtype(entries))
        for i, f in enumerate(fields):
            unified[f'{i}'] = f

        uniques = np.unique(unified)
        results = [uniques[f'{i}'] for i in range(len(fields))]
        return results


    def get_spans(self, field=None, fields=None):
        if field is None and fields is None:
            raise ValueError("One of 'field' and 'fields' must be set")
        if field is not None and fields is not None:
            raise ValueError("Only one of 'field' and 'fields' may be set")
        raw_field = None
        raw_fields = None
        if field is not None:
            per._check_is_reader_or_ndarray('field', field)
            raw_field = field[:] if isinstance(field, per.Reader) else field
        else:
            raw_fields = []
            for f in fields:
                per._check_is_reader_or_ndarray('elements of tuple/list fields', f)
                raw_fields.append(f[:] if isinstance(f, per.Reader) else f)
        return per._get_spans(raw_field, raw_fields)


    # TODO: needs a predicate to break ties: first, last?
    def apply_spans_index_of_min(self, spans, reader, writer=None):
        per._check_is_reader_or_ndarray('reader', reader)
        per._check_is_reader_or_ndarray_if_set('writer', reader)

        if isinstance(reader, per.Reader):
            raw_reader = reader[:]
        else:
            raw_reader = reader

        if writer is not None:
            if isinstance(writer, per.Writer):
                raw_writer = writer[:]
            else:
                raw_writer = writer
        else:
            raw_writer = np.zeros(len(spans)-1, dtype=np.int64)
            per._apply_spans_index_of_min(spans, raw_reader, raw_writer)
        if isinstance(writer, per.Writer):
            writer.write(raw_writer)
        else:
            return raw_writer


    def apply_spans_index_of_max(self, spans, reader, writer=None):
        per._check_is_reader_or_ndarray('reader', reader)
        per._check_is_reader_or_ndarray_if_set('writer', reader)

        if isinstance(reader, per.Reader):
            raw_reader = reader[:]
        else:
            raw_reader = reader

        if writer is not None:
            if isinstance(writer, per.Writer):
                raw_writer = writer[:]
            else:
                raw_writer = writer
        else:
            raw_writer = np.zeros(len(spans)-1, dtype=np.int64)
            per._apply_spans_index_of_max(spans, raw_reader, raw_writer)
        if isinstance(writer, per.Writer):
            writer.write(raw_writer)
        else:
            return raw_writer


    # TODO - for all apply_spans methods, spans should be able to be an ndarray
    def apply_spans_count(self, spans, _, writer):
        if isinstance(writer, per.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            per._apply_spans_count(spans, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            per._apply_spans_count(spans, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")

    def apply_spans_first(self, spans, reader, writer):
        if isinstance(reader, per.Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, per.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            per._apply_spans_first(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            per._apply_spans_first(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_last(self, spans, reader, writer):
        if isinstance(reader, per.Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, per.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            per._apply_spans_last(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(writer, np.ndarray):
            per._apply_spans_last(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_min(self, spans, reader, writer):
        if isinstance(reader, per.Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, per.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            per._apply_spans_min(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(reader, per.Reader):
            per._apply_spans_min(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_max(self, spans, reader, writer):
        if isinstance(reader, per.Reader):
            read_values = reader[:]
        elif isinstance(reader.np.ndarray):
            read_values = reader
        else:
            raise ValueError(f"'reader' must be one of 'Reader' or 'ndarray' but is {type(reader)}")

        if isinstance(writer, per.Writer):
            dest_values = writer.chunk_factory(len(spans) - 1)
            per._apply_spans_max(spans, read_values, dest_values)
            writer.write(dest_values)
        elif isinstance(reader, per.Reader):
            per._apply_spans_max(spans, read_values, writer)
        else:
            raise ValueError(f"'writer' must be one of 'Writer' or 'ndarray' but is {type(writer)}")


    def apply_spans_concat(self, spans, reader, writer):
        if not isinstance(reader, per.IndexedStringReader):
            raise ValueError(f"'reader' must be one of 'IndexedStringReader' but is {type(reader)}")
        if not isinstance(writer, per.IndexedStringWriter):
            raise ValueError(f"'writer' must be one of 'IndexedStringWriter' but is {type(writer)}")

        src_index = reader.field['index'][:]
        src_values = reader.field['values'][:]
        dest_index = np.zeros(reader.chunksize, src_index.dtype)
        dest_values = np.zeros(reader.chunksize * 16, src_values.dtype)

        max_index_i = reader.chunksize
        max_value_i = reader.chunksize * 8
        s = 0
        while s < len(spans) - 1:
            s, index_i, index_v = per._apply_spans_concat(spans, src_index, src_values,
                                                          dest_index, dest_values,
                                                          max_index_i, max_value_i, s)

            if index_i > 0 or index_v > 0:
                writer.write_raw(dest_index[:index_i], dest_values[:index_v])
        writer.flush()


    def aggregate_count(self, fkey_indices=None, fkey_index_spans=None,
                        reader=None, writer=None):
        return per._aggregate_impl(self.apply_spans_count, fkey_indices, fkey_index_spans,
                                   reader, writer, np.uint32)


    def aggregate_first(self, fkey_indices=None, fkey_index_spans=None,
                        reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_first, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_last(self, fkey_indices=None, fkey_index_spans=None,
                       reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_last, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_min(self,fkey_indices=None, fkey_index_spans=None,
                      reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_min, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_max(self, fkey_indices=None, fkey_index_spans=None,
                      reader=None, writer=None):
        return self.aggregate_custom(self.apply_spans_max, fkey_indices, fkey_index_spans,
                                     reader, writer)


    def aggregate_custom(self,
                         predicate, fkey_indices=None, fkey_index_spans=None,
                         reader=None, writer=None):
        if reader is None:
            raise ValueError("'reader' must not be None")
        if not isinstance(reader, (per.Reader, np.ndarray)):
            raise ValueError(f"'reader' must be a Reader or an ndarray but is {type(reader)}")
        if isinstance(reader, per.Reader):
            required_dtype = reader.dtype()
        else:
            required_dtype = reader.dtype
        return per._aggregate_impl(predicate, fkey_indices, fkey_index_spans,
                                   reader, writer, required_dtype)


    def join(self,
             destination_pkey, fkey_indices, values_to_join,
             writer=None, fkey_index_spans=None):
        if fkey_indices is not None:
            if not isinstance(fkey_indices, (per.Reader, np.ndarray)):
                raise ValueError(f"'fkey_indices' must be a type of Reader or an ndarray")
        if values_to_join is not None:
            if not isinstance(values_to_join, (per.Reader, np.ndarray)):
                raise ValueError(f"'values_to_join' must be a type of Reader but is {type(values_to_join)}")
            if isinstance(values_to_join, per.IndexedStringReader):
                raise ValueError(f"Joins on indexed string fields are not supported")

        if isinstance(values_to_join, per.Reader):
            raw_values_to_join = values_to_join[:]
        else:
            raw_values_to_join = values_to_join

        # generate spans for the sorted key indices if not provided
        if fkey_index_spans is None:
            fkey_index_spans = self.get_spans(field=fkey_indices)

        # select the foreign keys from the start of each span to get an ordered list
        # of unique id indices in the destination space that the results of the predicate
        # execution are mapped to
        unique_fkey_indices = fkey_indices[:][fkey_index_spans[:-1]]

        # generate a filter to remove invalid foreign key indices (where values in the
        # foreign key don't map to any values in the destination space
        invalid_filter = unique_fkey_indices < per.INVALID_INDEX
        safe_unique_fkey_indices = unique_fkey_indices[invalid_filter]

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_values_to_join = raw_values_to_join[invalid_filter]

        # now get the memory that the results will be mapped to
        destination_space_values = writer.chunk_factory(len(destination_pkey))

        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_values_to_join

        if writer is not None:
            writer.write(destination_space_values)
        else:
            return destination_space_values


    def predicate_and_join(self,
                           predicate, destination_pkey, fkey_indices,
                           reader=None, writer=None, fkey_index_spans=None):
        if reader is not None:
            if not isinstance(reader, per.Reader):
                raise ValueError(f"'reader' must be a type of Reader but is {type(reader)}")
            if isinstance(reader, per.IndexedStringReader):
                raise ValueError(f"Joins on indexed string fields are not supported")

        # generate spans for the sorted key indices if not provided
        if fkey_index_spans is None:
            fkey_index_spans = self.get_spans(field=fkey_indices)

        # select the foreign keys from the start of each span to get an ordered list
        # of unique id indices in the destination space that the results of the predicate
        # execution are mapped to
        unique_fkey_indices = fkey_indices[:][fkey_index_spans[:-1]]

        # generate a filter to remove invalid foreign key indices (where values in the
        # foreign key don't map to any values in the destination space
        invalid_filter = unique_fkey_indices < per.INVALID_INDEX
        safe_unique_fkey_indices = unique_fkey_indices[invalid_filter]

        # execute the predicate (note that not every predicate requires a reader)
        if reader is not None:
            dtype = reader.dtype()
        else:
            dtype = np.uint32
        results = np.zeros(len(fkey_index_spans)-1, dtype=dtype)
        predicate(fkey_index_spans, reader, results)

        # the predicate results are in the same space as the unique_fkey_indices, which
        # means they may still contain invalid indices, so filter those now
        safe_results = results[invalid_filter]

        # now get the memory that the results will be mapped to
        destination_space_values = writer.chunk_factory(len(destination_pkey))
        # finally, map the results from the source space to the destination space
        destination_space_values[safe_unique_fkey_indices] = safe_results

        writer.write(destination_space_values)


    def get_reader(self, field):
        if 'fieldtype' not in field.attrs.keys():
            raise ValueError(f"'{field_name}' is not a well-formed field")

        fieldtype_map = {
            'indexedstring': per.IndexedStringReader,
            'fixedstring': per.FixedStringReader,
            'categorical': per.CategoricalReader,
            'boolean': per.NumericReader,
            'numeric': per.NumericReader,
            'datetime': per.TimestampReader,
            'date': per.TimestampReader,
            'timestamp': per.TimestampReader
        }

        fieldtype = field.attrs['fieldtype'].split(',')[0]
        return fieldtype_map[fieldtype](self, field)


    def get_existing_writer(self, field, timestamp=None):
        if 'fieldtype' not in field.attrs.keys():
            raise ValueError(f"'{field_name}' is not a well-formed field")

        fieldtype_map = {
            'indexedstring': per.IndexedStringReader,
            'fixedstring': per.FixedStringReader,
            'categorical': per.CategoricalReader,
            'boolean': per.NumericReader,
            'numeric': per.NumericReader,
            'datetime': per.TimestampReader,
            'date': per.TimestampReader,
            'timestamp': per.TimestampReader
        }

        fieldtype = field.attrs['fieldtype'].split(',')[0]
        reader = fieldtype_map[fieldtype](self, field)
        group = field.parent
        name = field.name.split('/')[-1]
        return reader.get_writer(group, name, timestamp=timestamp, write_mode='overwrite')


    def get_indexed_string_writer(self, group, name, timestamp=None, writemode='write'):
        return per.IndexedStringWriter(self, group, name, timestamp, writemode)


    def get_fixed_string_writer(self, group, name, width,
                                timestamp=None, writemode='write'):
        return per.FixedStringWriter(self, group, name, width, timestamp, writemode)


    def get_categorical_writer(self, group, name, categories,
                               timestamp=None, writemode='write'):
        return per.CategoricalWriter(self, group, name, categories, timestamp, writemode)


    def get_numeric_writer(self, group, name, dtype, timestamp=None, writemode='write'):
        return per.NumericWriter(self, group, name, dtype, timestamp, writemode)


    def get_timestamp_writer(self, group, name, timestamp=None, writemode='write'):
        return per.TimestampWriter(self, group, name, timestamp, writemode)


    def get_compatible_writer(self, field, dest_group, dest_name,
                              timestamp=None, writemode='write'):
        reader = self.get_reader(field)
        return reader.get_writer(dest_group, dest_name, timestamp, writemode)


    def get_or_create_group(self, group, name):
        if name in group:
            return group[name]
        return group.create_group(name)


    def chunks(self, length, chunksize=None):
        if chunksize is None:
            chunksize = self.chunksize
        cur = 0
        while cur < length:
            next = min(length, cur + chunksize)
            yield cur, next
            cur = next

    def process(self, inputs, outputs, predicate):

        # TODO: modifying the dictionaries in place is not great
        input_readers = dict()
        for k, v in inputs.items():
            if isinstance(v, per.Reader):
                input_readers[k] = v
            else:
                input_readers[k] = self.get_reader(v)
        output_writers = dict()
        output_arrays = dict()
        for k, v in outputs.items():
            if isinstance(v, per.Writer):
                output_writers[k] = v
            else:
                raise ValueError("'outputs': all values must be 'Writers'")

        reader = next(iter(input_readers.values()))
        input_length = len(reader)
        writer = next(iter(output_writers.values()))
        chunksize = writer.chunksize
        required_chunksize = min(input_length, chunksize)
        for k, v in outputs.items():
            output_arrays[k] = output_writers[k].chunk_factory(required_chunksize)

        for c in self.chunks(input_length, chunksize):
            kwargs = dict()

            for k, v in inputs.items():
                kwargs[k] = v[c[0]:c[1]]
            for k, v in output_arrays.items():
                kwargs[k] = v[:c[1] - c[0]]
            predicate(**kwargs)

            # TODO: write back to the writer
            for k in output_arrays.keys():
                output_writers[k].write_part(kwargs[k])
        for k, v in output_writers.items():
            output_writers[k].flush()


    def get_index(self, target, foreign_key, destination=None):
        print('  building patient_id index')
        t0 = time.time()
        target_lookup = dict()
        for i, v in enumerate(target[:]):
            target_lookup[v] = i
        print(f'  target lookup built in {time.time() - t0}s')

        print('  perform initial index')
        t0 = time.time()
        foreign_key_elems = foreign_key[:]
        # foreign_key_index = np.asarray([target_lookup.get(i, -1) for i in foreign_key_elems],
        #                                    dtype=np.int64)
        foreign_key_index = np.zeros(len(foreign_key_elems), dtype=np.int64)

        current_invalid = np.int64(per.INVALID_INDEX)
        for i_k, k in enumerate(foreign_key_elems):
            index = target_lookup.get(k, current_invalid)
            if index >= per.INVALID_INDEX:
                current_invalid += 1
                target_lookup[k] = index
            foreign_key_index[i_k] = index
        print(f'  initial index performed in {time.time() - t0}s')

        if destination:
            destination.write(foreign_key_index)
        else:
            return foreign_key_index


    def get_trash_group(self, group):

        group_names = group.name[1:].split('/')

        while True:
            id = str(uuid.uuid4())
            try:
                result = group.create_group(f"/trash/{'/'.join(group_names[:-1])}/{id}")
                return result
            except KeyError:
                pass


    def temp_filename(self):
        uid = str(uuid.uuid4())
        while os.path.exists(uid + '.hdf5'):
            uid = str(uuid.uuid4())
        return uid + '.hdf5'


    def map_left(self, left_on, right_on, map_mode='both'):
        valid_map_modes = ('map_to', 'map_from', 'both')

        if map_mode not in valid_map_modes:
            raise ValueError("'map_mode' must be one of {}".format(valid_map_modes))

        l_key_raw = per._raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = per._raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=l_df, right=r_df, left_on='l_k', right_on='r_k', how='left')
        l_to_r_map = df['l_index'].to_numpy()
        l_to_r_filt = np.logical_not(df['l_index'].isnull())
        r_to_l_map = df['r_index'].to_numpy(dtype=np.int64)
        r_to_l_filt = np.logical_not(df['r_index'].isnull())
        return l_to_r_map, l_to_r_filt, r_to_l_map, r_to_l_filt

    def map_right(self, left_on, right_on, map_mode='both'):
        l_key_raw = per._raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = per._raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=r_df, right=l_df, left_on='r_k', right_on='l_k', how='left')
        l_to_r_map = df['r_index'].to_numpy()
        l_to_r_filt = np.logical_not(df['r_index'].isnull())
        r_to_l_map = df['l_index'].to_numpy(dtype=np.int64)
        r_to_l_filt = np.logical_not(df['l_index'].isnull())
        return l_to_r_map, l_to_r_filt, r_to_l_map, r_to_l_filt


    def merge(self, left_on, right_on, how,
              left_fields=None, right_fields=None,
              left_writers=None, right_writers=None):
        if how == 'left':
            self.merge_left(left_on, right_on, left_fields, right_fields,
                            left_writers, right_writers)
        elif how == 'right':
            self.merge_right(left_on, right_on, left_fields, right_fields,
                             left_writers, right_writers)
        elif how == 'inner':
            self.merge_inner(left_on, right_on, left_fields, right_fields,
                             left_writers, right_writers)
        elif how == 'outer':
            self.merge_outer(left_on, right_on, left_fields, right_fields,
                             left_writers, right_writers)


    def merge_left(self, left_on, right_on,
                   left_fields=None, right_fields=None,
                   left_writers=None, right_writers=None):
        l_key_raw = per._raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = per._raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=l_df, right=r_df, left_on='l_k', right_on='r_k', how='left')
        l_to_r_map = df['l_index'].to_numpy()
        l_to_r_filt = np.logical_not(df['l_index'].isnull()).to_numpy()
        r_to_l_map = df['r_index'].to_numpy(dtype=np.int64)
        r_to_l_filt = np.logical_not(df['r_index'].isnull()).to_numpy()
        print(df)

        print("l_to_r_map:", l_to_r_map, l_to_r_map.dtype)
        print("r_to_l_map:", r_to_l_map, r_to_l_map.dtype)

        # print(r_to_l_map > -1)
        left_results = list()
        for ilf, lf in enumerate(left_fields):
            lf_raw = per._raw_array_from_parameter(self, 'left_fields[{}]'.format(ilf), lf)
            joined_field = per._safe_map(lf_raw, l_to_r_map, l_to_r_filt)
            print(joined_field)
            if left_writers == None:
                left_results.append(joined_field)
            else:
                left_writers[ilf].write(joined_field)

        right_results = list()
        for irf, rf in enumerate(right_fields):
            rf_raw = per._raw_array_from_parameter(self, 'right_fields[{}]'.format(irf), rf)
            joined_field = per._safe_map(rf_raw, r_to_l_map, r_to_l_filt)
            print(joined_field)
            if right_writers == None:
                right_results.append(joined_field)
            else:
                right_writers[irf].write(joined_field)

        if left_writers == None:
            return left_results, right_results


    def merge_right(self, left_on, right_on,
                    left_fields=None, right_fields=None,
                    left_writers=None, right_writers=None):
        l_key_raw = per._raw_array_from_parameter(self, 'left_on', left_on)
        l_index = np.arange(len(l_key_raw), dtype=np.int64)
        l_df = pd.DataFrame({'l_k': l_key_raw, 'l_index': l_index})

        r_key_raw = per._raw_array_from_parameter(self, 'right_on', right_on)
        r_index = np.arange(len(r_key_raw), dtype=np.int64)
        r_df = pd.DataFrame({'r_k': r_key_raw, 'r_index': r_index})

        df = pd.merge(left=r_df, right=l_df, left_on='r_k', right_on='l_k', how='left')
        l_to_r_map = df['r_index'].to_numpy()
        l_to_r_filt = np.logical_not(df['r_index'].isnull())
        r_to_l_map = df['l_index'].to_numpy(dtype=np.int64)
        r_to_l_filt = np.logical_not(df['l_index'].isnull())
        print(df)

        print("l_to_r_map:", l_to_r_map, l_to_r_map.dtype)
        print("r_to_l_map:", r_to_l_map, r_to_l_map.dtype)

        print(r_to_l_map > -1)
        left_results = list()
        for ilf, lf in enumerate(left_fields):
            joined_field = per._safe_map(lf, r_to_l_map)
            print(joined_field)
            if left_writers == None:
                left_results.append(joined_field)
            else:
                left_writers[ilf].write(joined_field)

        right_results = list()
        for irf, rf in enumerate(right_fields):
            joined_field = per._safe_map(rf, l_to_r_map)
            print(joined_field)
            if right_writers == None:
                right_results.append(joined_field)
            else:
                right_writers[irf].write(joined_field)


    def merge_inner(self, left_on, left_fields, right_on, right_fields):
        raise NotImplementedError()


    def merge_outer(self, left_on, left_fields, right_on, right_fields):
        raise NotImplementedError()
