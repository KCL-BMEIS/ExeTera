from typing import Mapping
from exetera.core.abstract_types import DataFrame
import numpy as np
from exetera.core import fields as fld
from exetera.core import operations as ops
from exetera.core.data_writer import DataWriter
from exetera.core import utils
from datetime import datetime, date, timezone
import pytz

INDEXED_STRING_FIELD_SIZE = 10 # guessing

FIELD_MAPPING_TO_IMPORTER = {
    'categorical': lambda categories, value_type, allow_freetext:
                   lambda s, df, name, ts: LeakyCategoricalImporter(s, df, name, categories, value_type, ts) if allow_freetext else CategoricalImporter(s, df, name, categories, value_type, ts),
    'numeric': lambda dtype, invalid_value, validation_mode, create_flag_field, flag_field_name:
               lambda s, df, name, ts: NumericImporter(s, df, name, dtype, invalid_value, validation_mode, create_flag_field, flag_field_name, timestamp=ts),
    'string': lambda fixed_length:
              lambda s, df, name, ts: IndexedStringImporter(s, df, name, ts) if fixed_length is None else FixedStringImporter(s, df, name, fixed_length, ts),
    'datetime': lambda create_day_field, create_flag_field:
                lambda s, df, name, ts: DateTimeImporter(s, df, name, create_day_field, create_flag_field, ts),
    'date': lambda create_day_field, create_flag_field:
            lambda s, df, name, ts: DateImporter(s, df, name, create_day_field, create_flag_field, ts),
}

#========= ImporterDefinition, include Categorical, Numeric, String, Datetime, Date =========

class ImporterDefinition:
    def __init__(self):
        self._field_size = 0
        self._importer = None


class Categorical(ImporterDefinition):
    """
    Categorical is an importer definition for categorical fields. It's the means that users define categorical field in the schema dictionary.
    e.g. (1) Categorical(categories={"": 0, "NW1":1, "E1":2, "SW1P":3, "NW3":4})
         (2) Categorical(categories={"":0, "bachelor":1, "master":2, "doctor":3}, allow_freetext=True)

    :param categories: dictionary that contain key/value pair for Categorical Field
    :param value_type: value type in the dictionary. Default is 'int8'.
    :param allow_freetext: If allow_freetext is True, will create extra column "**_freetext" for unexpected text.
    """
    def __init__(self, categories, value_type='int8', allow_freetext=False):
        """
        Create categorical importer definition.
        """
        self._field_size = max([len(k) for k in categories.keys()])

        self._importer = FIELD_MAPPING_TO_IMPORTER['categorical'](categories, value_type, allow_freetext)


class Numeric(ImporterDefinition):
    """
    Numeric is an importer definition for numeric fields. It's the means that users define numeric field in the schema dictionary.
    e.g. Numeric('float32', invalid_value = 160.5, validation_mode='relaxed', flag_field_name= '_valid_test')

    :param dtype: datatype. The admitted datatype is as following: 'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 
                  'uint32', 'uint64', 'float', 'float32', 'float64', 'bool'
    :param invalid_value: replace the data with invalid_value when the data is invalid. The default is 0.
    :param validation_mode: three validation mode: "strict", "allow_empty", "relaxed". The default is "allow_empty"
    :param create_flag_field: create extra field which indicate if the data is valid or not. The default is True.
    :param flag_field_name: the suffix for the flag field. The default is "_valid".
    """
    def __init__(self, dtype, invalid_value=0, validation_mode='allow_empty', create_flag_field='True', flag_field_name='_valid'):
        if dtype in ('int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
            self._field_size = 20
        elif dtype in ('float', 'float32', 'float64'):
            self._field_size = 30
        elif dtype == 'bool':
            self._field_size = 5
        else:
            raise ValueError("Unrecognised numeric type '{}' in the field".format(dtype))

        self._importer = FIELD_MAPPING_TO_IMPORTER['numeric'](dtype, invalid_value, validation_mode, create_flag_field, flag_field_name)


class String(ImporterDefinition):
    """
    String is an importer definition for string fields. It's the means that users define string field in the schema dictionary.
    e.g. (1) String()
         (2) String(fixed_length=4)

    :param fixed_length: set the fixed_length if the field type is fixed string.
    """
    def __init__(self, fixed_length: int = None):
        if fixed_length:
            self._field_size = fixed_length
        else:
            self._field_size = INDEXED_STRING_FIELD_SIZE 

        self._importer = FIELD_MAPPING_TO_IMPORTER['string'](fixed_length)


class DateTime(ImporterDefinition):
    """
    DateTime is an importer definition for DateTime fields. It's the means that users define DateTime field in the schema dictionary.
    e.g. DateTime(create_day_field=True)

    :param create_day_field: create extra field which contains the date information.
    :param create_flag_field: create extra field which indicate if the data is valid or not. The default is True.
    """
    def __init__(self, create_day_field=False, create_flag_field=False):
        self._field_size = 32
        self._importer = FIELD_MAPPING_TO_IMPORTER['datetime'](create_day_field, create_flag_field)


class Date(ImporterDefinition):
    """
    Date is an importer definition for Date fields. It's the means that users define Date field in the schema dictionary.
    e.g. Date(create_day_field=True)

    :param create_day_field: create extra field which contains the date information.
    :param create_flag_field: create extra field which indicate if the data is valid or not. The default is True.
    """
    def __init__(self, create_day_field=False, create_flag_field=False):
        self._field_size = 10
        self._importer = FIELD_MAPPING_TO_IMPORTER['date'](create_day_field, create_flag_field)


#============= Field Importers ============

class CategoricalImporter:
    """
    Importer for Categorical field, a private attribute defined in Categorical ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this categorical field.
    :param categories: dictionary that contain key/value pair.
    :param value_type: value type in the dictionary. Default is 'int8'.
    :param timestamp: timestamp for creating this categorical field. Default is None.
    """
    def __init__(self, session, df:DataFrame, name:str, categories:Mapping[str, str], value_type:str='int8', timestamp=None):
        if not isinstance(categories, dict):
            raise ValueError("'categories' must be of type dict but is {} in the field '{}'".format(type(categories), name))
        elif len(categories) == 0:
            raise ValueError("'categories' must not be empty in the field '{}'".format(name))

        self.field = df.create_categorical(name, value_type, categories, timestamp, None)
        self.byte_map = ops.get_byte_map(categories)
        self.field_size = max([len(k) for k in categories])

    def import_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        chunk = np.zeros(written_row_count, dtype=np.uint8)
        cat_keys, cat_index, cat_values = self.byte_map
                
        ops.categorical_transform(chunk, col_idx, column_inds, column_vals, column_offsets, cat_keys, cat_index, cat_values)
        self.field.data.write_part(chunk)

    def complete(self):
        self.field.data.complete()


class LeakyCategoricalImporter:
    """
    Importer for LeakyCategoricalField when allow_freetext is True, a private attribute defined in Categorical ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this leaky categorical field.
    :param categories: dictionary that contain key/value pair.
    :param value_type: value type in the dictionary. Default is 'int8'.
    :param timestamp: timestamp for creating this leaky categorical field. Default is None.
    """
    def __init__(self, session, df:DataFrame, name:str, categories:Mapping[str, str],
                       value_type:str='int8', timestamp=None):
        self.byte_map = ops.get_byte_map(categories)
        self.freetext_index_accumulated = 0
        self.field = df.create_categorical(name, value_type, categories, timestamp, None)
        self.other_values_field = df.create_indexed_string(f"{name}_freetext", timestamp, None)
        self.other_values_field.indices.write_part([0])


    def import_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
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
        self.other_values_field.indices.write_part(freetext_indices[1:])
        self.other_values_field.values.write_part(freetext_values)


    def complete(self):
        # add a 'freetext' value to keys
        self.field.keys['freetext'] = -1
        self.field.data.complete()
        self.other_values_field.data.complete()
        

class NumericImporter:
    """
    Importer for Numeric field, a private attribute defined in Numeric ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this numeric field.
    :param dtype: datatype. The admitted datatype is as following: 'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16',
                  'uint32', 'uint64', 'float', 'float32', 'float64', 'bool'
    :param invalid_value: replace the data with invalid_value when the data is invalid. The default is 0.
    :param validation_mode: three validation mode: "strict", "allow_empty", "relaxed". The default is "allow_empty"
    :param create_flag_field: create extra field which indicate if the data is valid or not. The default is True.
    :param flag_field_name: the suffix for the flag field. The default is "_valid".
    :param timestamp: timestamp for creating this numeric field. Default is None.
    """
    def __init__(self, session, df:DataFrame, name:str, dtype:str, invalid_value=0,
                       validation_mode='allow_empty', create_flag_field=True, flag_field_suffix='_valid',
                       timestamp=None):
        self.field = df.create_numeric(name, dtype, timestamp, None)

        create_flag_field = create_flag_field if validation_mode in ('allow_empty', 'relaxed') else False
        self.flag_field = None
        if create_flag_field:
            self.flag_field = df.create_numeric(f"{name}{flag_field_suffix}", 'bool', timestamp, None)

        self.dtype = dtype
        self.field_name = name
        self.invalid_value = invalid_value
        self.validation_mode = validation_mode

        if isinstance(invalid_value, str) and invalid_value.strip() in ('min', 'max'):
            if dtype == 'bool':
                raise ValueError('Field {} is bool type. It should not have min/max as default value')
            else:
                (min_value, max_value) = utils.get_min_max(dtype)
                self.invalid_value = min_value if invalid_value.strip() == 'min' else max_value


    def import_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
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

            if exception_message != 0:
                ops.raiseNumericException(exception_message, exception_args)

        elif self.dtype in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64') :
            elements, validity = ops.transform_int(
                column_inds, column_vals, column_offsets, col_idx,
                written_row_count, self.invalid_value, self.validation_mode,
                value_dtype, self.field_name)
        else:
            elements, validity = ops.transform_float(
                column_inds, column_vals, column_offsets, col_idx,
                written_row_count, self.invalid_value, self.validation_mode,
                value_dtype, self.field_name)

        self.field.data.write_part(elements)
        if self.flag_field is not None:
            self.flag_field.data.write_part(validity)


    def _is_blank(self, value):
        return (isinstance(value, str) and value.strip() == '') or value == b''


    def complete(self):
        self.field.data.complete()
        if self.flag_field is not None:
            self.flag_field.data.complete()


class IndexedStringImporter:
    """
    Importer for Indexed String field, a private attribute defined in String ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this indexed string field.
    :param timestamp: timestamp for creating this indexed string field. Default is None.
    """
    def __init__(self, session, df, name, timestamp=None):
        self.field = df.create_indexed_string(name, timestamp, None)
        self.chunk_accumulated = 0
        self.field.indices.write_part([0])

    def import_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        # broadcast accumulated size to current index array
        index = column_inds[col_idx, :written_row_count + 1] + self.chunk_accumulated
        self.chunk_accumulated += column_inds[col_idx, written_row_count]

        col_offset = column_offsets[col_idx]
        values = column_vals[col_offset: col_offset + column_inds[col_idx, written_row_count]]
        self.write_part(index, values)

    def write_part(self, index, values):
        if index.dtype != np.int64:
            raise ValueError(f"'index' must be an ndarray of '{np.int64}'")
        if values.dtype not in (np.uint8, 'S1'):
            raise ValueError(f"'values' must be an ndarray of '{np.uint8}' or 'S1'")
        self.field.indices.write_part(index[1:])
        self.field.values.write_part(values)

    def complete(self):
        # self.field.data.complete()
        self.field.indices.complete()
        self.field.values.complete()


class FixedStringImporter:
    """
    Importer for Fixed String field, a private attribute defined in String ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this fixed string field.
    :param strlen: set the fixed length for this fixed string field.
    :param timestamp: timestamp for creating this fixed string field. Default is None.
    """
    def __init__(self, session, df, name, strlen, timestamp = None):
        self.field = df.create_fixed_string(name, strlen, timestamp, None)  
        self.strlen = strlen
        self.field_size = strlen        

    def import_part(self, column_inds, column_vals, column_offsets,  col_idx, written_row_count):
        values = np.zeros(written_row_count, dtype='S{}'.format(self.strlen))
        ops.fixed_string_transform(column_inds, column_vals, column_offsets, col_idx,
                                   written_row_count, self.strlen, values.data.cast('b'))
        self.field.data.write_part(values)

    def complete(self):
        self.field.data.complete()


class DateTimeImporter:
    """
    Importer for DateTime field, a private attribute defined in DateTime ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this datetime field.
    :param create_day_field: create extra field which contains the date information.
    :param create_flag_field: create extra field which indicate if the data is valid or not. The default is True.
    :param timestamp: timestamp for creating this datetime field. Default is None.
    """
    def __init__(self, session, df, name, create_day_field=False, create_flag_field=False, timestamp=None):
        self.field = df.create_timestamp(name, timestamp, None)   
        self.day_field = None
        if create_day_field:
            self.day_field = df.create_fixed_string(f"{name}_day", 10, timestamp, None)  

        self.flag_field = None
        if create_flag_field:
            self.flag_field = df.create_numeric(f"{name}_set", 'bool', timestamp, None)

    def write_part(self, values):
        datetime_ts = np.zeros(len(values), dtype=np.float64)
        dates = np.zeros(len(values),dtype='S10')
        flags = np.ones(len(values), dtype='bool')

        for i in range(len(values)):
            value = values[i].strip()
            if value == b'':
                datetime_ts[i] = 0
                flags[i] = False
            else:
                v_len = len(value)
                if v_len == 32:
                    # ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S.%f%z')
                    v_datetime = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                          int(value[11:13]), int(value[14:16]), int(value[17:19]),
                                          int(value[20:26]), tzinfo=timezone.utc)
                elif v_len == 26:  # b'2020-06-15 19:45:39.05 UTC'
                    v_datetime = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                          int(value[11:13]), int(value[14:16]), int(value[17:19]),
                                          int(value[20:22]) * 10000)
                elif v_len == 25:
                    # ts = datetime.strptime(value.decode(), '%Y-%m-%d %H:%M:%S%z')
                    v_datetime = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                          int(value[11:13]), int(value[14:16]), int(value[17:19]), tzinfo=timezone.utc)
                elif v_len == 19:
                    v_datetime = datetime(int(value[0:4]), int(value[5:7]), int(value[8:10]),
                                          int(value[11:13]), int(value[14:16]), int(value[17:19]), tzinfo=timezone.utc)
                else:
                    raise ValueError(f"Date field '{self.field}' has unexpected format '{value}'")
                datetime_ts[i] = v_datetime.timestamp()
                dates[i] = value[:10]
        
        self.field.data.write_part(datetime_ts)
        if self.day_field is not None:
            self.day_field.data.write_part(dates)
        if self.flag_field is not None:
            self.flag_field.data.write_part(flags)

    def import_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        data = ops.transform_to_values(column_inds, column_vals, column_offsets, col_idx, written_row_count)
        data = [x.tobytes().strip() for x in data]
        self.write_part(data)

    def complete(self):
        self.field.data.complete()
        if self.day_field is not None:
            self.day_field.data.complete()
        if self.flag_field is not None:
            self.flag_field.data.complete()


class DateImporter:
    """
    Importer for Date field, a private attribute defined in Date ImporterDefinition.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this date field.
    :param create_day_field: create extra field which contains the date information.
    :param create_flag_field: create extra field which indicate if the data is valid or not. The default is True.
    :param timestamp: timestamp for creating this date field. Default is None.
    """
    def __init__(self, session, df, name, create_day_field=False, create_flag_field=False, timestamp=None):
        self.field = df.create_timestamp(name, timestamp, None)   
        self.day_field = None
        #print('create_day_field' , create_day_field)
        if create_day_field:
            self.day_field = df.create_fixed_string(f"{name}_day", 10, timestamp, None) 

        self.flag_field = None
        if create_flag_field:
            self.flag_field = df.create_numeric(f"{name}_set", 'bool', timestamp, None)
    
    def write_part(self, values):
        date_ts = np.zeros(len(values), dtype=np.float64)
        flags = np.ones(len(values), dtype='bool')

        for i in range(len(values)):
            value = values[i].strip()
            if value == b'':
                date_ts[i] = 0
                flags[i] = False
            else:
                ts = datetime.strptime(value.decode(), '%Y-%m-%d')
                ts = ts.replace(tzinfo=timezone.utc)
                date_ts[i] = ts.timestamp()

        self.field.data.write_part(date_ts)
        if self.day_field is not None:
            self.day_field.data.write_part(values)    
        if self.flag_field is not None:
            self.flag_field.data.write_part(flags)

    def import_part(self, column_inds, column_vals, column_offsets, col_idx, written_row_count):
        data = ops.transform_to_values(column_inds, column_vals, column_offsets, col_idx, written_row_count)
        data = [x.tobytes().strip() for x in data]
        self.write_part(data)

    def complete(self):
        self.field.data.complete()
        if self.day_field is not None:
            self.day_field.data.complete()
        if self.flag_field is not None:
            self.flag_field.data.complete()

class TimestampImporter:
    """
    Importer for Timestamp field, currently only used in parser.py to parse timestamp field.

    :param session: session. It will be deprecated in future version.
    :param df: source dataframe.
    :param name: field name of this timestamp field.
    :param timestamp: timestamp for creating this timestamp field. Default is None.
    """
    def __init__(self, session, df, name, timestamp=None):
        self.field = df.create_timestamp(name, timestamp, None)

    def write_part(self, values):
        self.field.data.write_part(values)

    def complete(self):
        self.field.data.complete()

    def write(self, values):
        self.write_part(values)
        self.complete()