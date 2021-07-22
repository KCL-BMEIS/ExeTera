from typing import Mapping
from exetera.core.abstract_types import DataFrame
import numpy as np
from exetera.core import fields as fld
from exetera.core import operations as ops
from exetera.core.data_writer import DataWriter


FIELD_MAPPING_TO_IMPORTER = {
    'categorical': lambda categories, value_type, allow_freetext:
                   lambda s, df, name, ts: LeakyCategoricalImporter(s, df, name, categories, value_type, ts) if allow_freetext else CategoricalImporter(s, df, name, categories, value_type, ts),
    'numeric': lambda dtype, invalid_value, validation_mode, create_flag_field, flag_field_name:
               lambda s, df, name, ts: NumericImporter(s, df, name, dtype, invalid_value, validation_mode, create_flag_field, flag_field_name, timestamp=ts),
    'string': lambda fixed_length:
              lambda s, df, name, ts: IndexedStringImporter(s, df, name, ts) if fixed_length is not None else FixedStringImporter(s, df, name, fixed_length, ts)
}

#===== Field Mapping Type, include Categorical, Numeric, , , Datetime =======
class FieldMappingType:
    def __init__(self):
        self.field_size = 0
        self.importer = None



class Categorical(FieldMappingType):
    def __init__(self, categories, value_type='int8', allow_freetext=False):
        """
        :param categories: dictionary that contain key/value pair for Categorical Field
        :param value_type: value type in the dictionary. Default is 'int8'.
        """
        self.field_size = max([len(k) for k in categories.keys()])

        self.importer = FIELD_MAPPING_TO_IMPORTER['categorical'](categories, value_type, allow_freetext)



class Numeric(FieldMappingType):
    def __init__(self, dtype, invalid_value=0, validation_mode='allow_empty', create_flag_field = 'True', flag_field_name='_valid'):
        if dtype in ('int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
            self.field_size = 20
        elif dtype in ('float', 'float32', 'float64'):
            self.field_size = 30
        elif dtype == 'bool':
            self.field_size = 5
        else:
            raise ValueError("Unrecognised numeric type '{}' in the field".format(dtype))

        self.importer = FIELD_MAPPING_TO_IMPORTER['numeric'](dtype, invalid_value, validation_mode, create_flag_field, flag_field_name)


class String(FieldMappingType):
    def __init__(self, fixed_length: int = None):
        if fixed_length:
            self.field_size = fixed_length
        else:
            self.field_size = 10 # guessing

        self.importer = FIELD_MAPPING_TO_IMPORTER['string'](fixed_length)


class DateTime(FieldMappingType):
    def __init__(self, create_day_field=False):
        self.field_size = 32

class Date(FieldMappingType):
    def __init__(self):
        self.field_size = 10


#============= Field Importers ============

class CategoricalImporter:
    def __init__(self, session, df:DataFrame, name:str, categories:Mapping[str, FieldMappingType],
                       value_type:str='int8', timestamp=None):
        if not isinstance(categories, dict):
            raise ValueError("'categories' must be of type dict but is {} in the field '{}'".format(type(categories, name)))
        elif len(categories) == 0:
            raise ValueError("'categories' must not be empty in the field '{}'".format(name))

        self.field = df.create_categorical(name, value_type, categories, timestamp, None)
        self.byte_map = ops.get_byte_map(categories)
        self.field_size = max([len(k) for k in categories])

    def write_part(self, values):
        self.field.data.write_part(values)

    def complete(self):
        self.field.data.complete()

    def transform_and_write_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        chunk = np.zeros(written_row_count, dtype=np.uint8)
        cat_keys, cat_index, cat_values = self.byte_map
                
        ops.categorical_transform(chunk, col_idx, column_inds, column_vals, column_offsets, cat_keys, cat_index, cat_values)
        self.field.data.write_part(chunk)


class LeakyCategoricalImporter:
    def __init__(self, session, df:DataFrame, name:str, categories:Mapping[str, FieldMappingType],
                       value_type:str='int8', timestamp=None):
 
        self.field = df.create_categorical(name, value_type, categories, timestamp, None)

        self.other_values_field = df.create_indexed_string(f"{name}_freetext", timestamp, None)

        self.byte_map = ops.get_byte_map(categories)
        self.freetext_index_accumulated = 0

    def write_part(self, values):
        results = np.zeros(len(values), dtype='int8')
        strresults = list([""] * len(values))
        keys = self.writer.keys
        anomalous_count = 0
        for i in range(len(values)):
            value = keys.get(values[i], -1)
            if value != -1:
                results[i] = value
            else:
                anomalous_count += 1
                results[i] = -1
                strresults[i] = values[i]
        self.field.data.write_part(results)
        self.other_values_field.data.write_part(strresults)

    def complete(self):
        # add a 'freetext' value to keys
        self.field.keys['freetext'] = -1
        self.field.data.complete()
        self.other_values_field.data.flush()

    def transform_and_write_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        cat_keys, cat_index, cat_values = self.byte_map
        chunk = np.zeros(written_row_count, dtype=np.int8) # use np.int8 instead of np.uint8, as we set -1 for leaky key
        freetext_indices_chunk = np.zeros(written_row_count + 1, dtype = np.int64)

        col_count = column_offsets[col_idx + 1] - column_offsets[col_idx]
        freetext_values_chunk = np.zeros(np.int64(col_count), dtype = np.uint8)

        ops.leaky_categorical_transform(chunk, freetext_indices_chunk, freetext_values_chunk, col_idx, column_inds, column_vals, column_offsets, cat_keys, cat_index, cat_values)

        freetext_indices = freetext_indices_chunk + self.freetext_index_accumulated # broadcast
        self.freetext_index_accumulated += freetext_indices_chunk[written_row_count]
        freetext_values = freetext_values_chunk[:freetext_indices_chunk[written_row_count]]
        self.field.data.write_part(chunk)
        self.other_values_field.data.write_part_raw(freetext_indices, freetext_values)
        

class NumericImporter:
    def __init__(self, session, df:DataFrame, name:str, dtype:str, invalid_value=0,
                 validation_mode='allow_empty', create_flag_field=True, flag_field_suffix='_valid',
                 timestamp=None):
        self.field = df.create_numeric(name, dtype, timestamp, None)

        self.flag_field = None
        if create_flag_field:
            self.flag_field = df.create_numeric(f"{name}{flag_field_suffix}", 'bool', timestamp, None)

        self.dtype = dtype
        self.field_name = name
        self.invalid_value = invalid_value
        self.validation_mode = validation_mode


    def transform_and_write_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        value_dtype = ops.str_to_dtype(self.dtype)

        if self.dtype == 'bool':
            # TODO: replace with fast reader based on categorical string parsing
            elements = np.zeros(written_row_count, dtype=self.dtype)
            validity = np.ones(written_row_count, dtype=bool)
            exception_message, exception_args = ops.numeric_bool_transform(
                elements, validity, column_inds, column_vals, column_offsets, col_idx,
                written_row_count, self.invalid_value,
                self.validation_mode, np.frombuffer(bytes(self.field_name, "utf-8"), dtype=np.uint8)
            )
        elif self.dtype in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64') :

            exception_message, exception_args = 0, []
            elements, validity = ops.transform_int_2(
                column_inds, column_vals, column_offsets, col_idx,
                written_row_count, self.invalid_value, self.validation_mode,
                value_dtype, self.field_name)
        else:
            exception_message, exception_args = 0, []
            elements, validity = ops.transform_float_2(
                column_inds, column_vals, column_offsets, col_idx,
                written_row_count, self.invalid_value, self.validation_mode,
                value_dtype, self.field_name)

        if exception_message != 0:
            ops.raiseNumericException(exception_message, exception_args)

        self.field.data.write_part(elements)
        if self.flag_field is not None:
            self.flag_field.data.write_part(validity)


    def _is_blank(self, value):
        return (isinstance(value, str) and value.strip() == '') or value == b''



class IndexedStringImporter:
    def __init__(self, session, df, name, timestamp=None):
        self.field = df.create_indexed_string(name, timestamp, None)
        self.chunk_accumulated = 0
        

        if 'index' not in self.field.keys(): 
            DataWriter.write(self.field, 'index', [0], 1)

    def write_part(self, index, values):
        if index.dtype != np.int64:
            raise ValueError(f"'index' must be an ndarray of '{np.int64}'")
        if values.dtype not in (np.uint8, 'S1'):
            raise ValueError(f"'values' must be an ndarray of '{np.uint8}' or 'S1'")
        DataWriter.write(self.field, 'index', index[1:], len(index)-1)
        DataWriter.write(self.field, 'values', values, len(values))

    def complete(self):
        self.field.data.complete()

    def transform_and_write_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        # broadcast accumulated size to current index array
        index = column_inds[col_idx, :written_row_count + 1] + self.chunk_accumulated
        self.chunk_accumulated += column_inds[col_idx, written_row_count]

        col_offset = column_offsets[col_idx]
        values = column_vals[col_offset: col_offset + column_inds[col_idx, written_row_count]]
        self.write_part(index, values)


class FixedStringImporter:
    def __init__(self, session, df, name, length, timestamp = None):
        self.field = df.create_fixed_string(name, length, timestamp, None)  
        self.length = length
        self.field_size = length


    def write_part(self, values):
        DataWriter.write(self.field, 'values', values, len(values))

    def transform_and_write_part(self, column_inds, column_vals, column_offsets,  col_idx, written_row_count):
        values = np.zeros(written_row_count, dtype='S{}'.format(self.strlen))
        ops.fixed_string_transform(column_inds, column_vals, column_offsets, col_idx,
                                     written_row_count, self.strlen, values.data.cast('b'))
        self.write_part(values)

    def complete(self):
        self.field.data.complete()

