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

import os
from datetime import datetime, MAXYEAR
from itertools import accumulate

import numpy as np
import h5py
import json
from io import StringIO

from exetera.core import persistence as per
from exetera.core import utils
from exetera.core import operations as ops
from exetera.core.csv_reader_speedup import read_file_using_fast_csv_reader
from exetera.core.field_importers import Categorical, Numeric, String, DateTime, Date
from exetera.core import validation as val
from exetera.core.session import Session
import exetera


def import_with_schema(session, timestamp, dataset_name, dest_file_name, schema_file, files, overwrite, include=None, exclude=None, chunk_row_size = 1 << 20):

    print(timestamp)
    print(schema_file)
    print(files)

    schema = None
    if isinstance(schema_file, str):
        with open(schema_file, encoding='utf-8') as sf:
            schema = load_schema(sf)
    elif isinstance(schema_file, StringIO):
        schema = load_schema(schema_file)

    any_parts_present = False
    for sk in schema.keys():
        if sk in files:
            any_parts_present = True
    if not any_parts_present:
        raise ValueError("none of the data sources in 'files' contain relevant data to the schema")

    # check if there's any table from the include/exclude doesn't exist in the input files
    input_file_tables = set(files.keys())
    include_tables = set(include.keys()) if include is not None else None
    exclude_tables = set(exclude.keys()) if exclude is not None else None
    if include_tables and not include_tables.issubset(input_file_tables):
        extra_tables = include_tables.difference(input_file_tables)
        raise ValueError("-n/--include: the following include table(s) are not part of any input files: {}".format(extra_tables))

    if exclude_tables and not exclude_tables.issubset(input_file_tables):
        extra_tables = exclude_tables.difference(input_file_tables)
        raise ValueError("-x/--exclude: the following exclude table(s) are not part of any input files: {}".format(extra_tables))

    stop_after = {}
    reserved_column_names = ('j_valid_from', 'j_valid_to')
    datastore = per.DataStore()

    if overwrite:
        mode = 'w'
    else:
        mode = 'r+'

    if isinstance(dest_file_name, str) and not os.path.exists(dest_file_name):
        mode = 'w'

        
    for sk in schema.keys():
        if sk in reserved_column_names:
            msg = "{} is a reserved column name: reserved names are {}"
            raise ValueError(msg.format(sk, reserved_column_names))

        schema_dict = schema[sk]
        csv_file = files[sk]

        include_fields = include.get(sk, None) if include is not None else None
        exclude_fields = exclude.get(sk, None) if exclude is not None else None

        print('dest_file_name', dest_file_name)
        ds = session.open_dataset(dest_file_name, mode, dataset_name)
        ddf = ds.create_dataframe(sk) 

        exetera.read_csv(csv_file, ddf, schema_dict, include_fields, exclude_fields, chunk_row_size)

  
        #     print(sk, hf.keys())
        #     table = hf[sk]
        #     ids = datastore.get_reader(table[list(table.keys())[0]])
        #     jvf = datastore.get_timestamp_writer(table, 'j_valid_from')
        #     ftimestamp = utils.string_to_datetime(timestamp).timestamp()
        #     valid_froms = np.full(len(ids), ftimestamp)
        #     jvf.write(valid_froms)
        #     jvt = datastore.get_timestamp_writer(table, 'j_valid_to')
        #     valid_tos = np.full(len(ids), ops.MAX_DATETIME.timestamp())
        #     jvt.write(valid_tos)

        # print(hf.keys())



def load_schema(source, verbosity=0):
    d = json.load(source)
    if verbosity > 1:
        print(d.keys())


    valid_versions = ('1.0.0', '1.1.0')

    if 'hystore' not in d.keys() and 'exetera' not in d.keys():
        raise ValueError("'{}' is not a valid ExeTera schema file".format(source))
    if 'hystore' in d.keys():
        if 'version' not in d['hystore']:
            raise ValueError("'version' field missing from 'hystore' top-level tag")
        elif d['hystore']['version'] != '1.0.0':
            raise ValueError("If the obsolete 'hystore' key is used, the version must be '1.0.0'")
    elif 'exetera' in d:
        if 'version' not in d['exetera']:
            raise ValueError("'version' field missing from 'exetera' top-level tag")
        elif d['exetera']['version'] not in valid_versions:
            msg = "The version number '{}' is not valid; it must be one of '{}'"
            raise ValueError(msg.format(d['exetera']['version'], valid_versions))

    if 'schema' not in d.keys():
        raise ValueError("'schema' top-level tag is missing from the schema file")

    schemas = d['schema']
    spaces = dict()
    for sk, sv in schemas.items():
        schema_dict = schema_file_to_dict(sk, sv)
        spaces[sk] = schema_dict
    return spaces


def schema_file_to_dict(sk, schema):
    permitted_numeric_types = ('float32', 'float64', 'bool', 'int8', 'uint8', 
                               'int16', 'uint16', 'int32', 'uint32', 'int64')

    fields = schema.get('fields', None)

    schema_dict = dict()

    for fk, fv in fields.items():
        val.validate_require_key(fk, 'field_type', fv)
        field_type = fv['field_type']

        if field_type == 'categorical':
            val.validate_require_key(fk, 'categorical', fv)
            categorical = fv['categorical']
                
            val.validate_require_key(fk, 'strings_to_values', categorical)
            strs_to_vals = categorical['strings_to_values']

            val.validate_require_key(fk, 'value_type', categorical)
            value_type = categorical['value_type']

            allow_freetext = True if 'out_of_range' in categorical else False
                
            importer_def = Categorical(strs_to_vals, value_type, allow_freetext)
            
        elif field_type == 'string':
            importer_def = String()

        elif field_type == 'fixed_string':
            val.validate_require_key(fk, 'length', fv)
            length = int(fv['length'])
            importer_def = String(fixed_length = length)
            
        elif field_type == 'numeric':
            val.validate_require_key(fk, 'value_type', fv)
            value_type = fv['value_type']

            if value_type not in permitted_numeric_types:
                msg = "Field {} has an invalid value_type '{}'. Permitted types are {}"
                raise ValueError(msg.format(fk, value_type, permitted_numeric_types))
        
            # default value for invalid numeric value
            invalid_value = 0
            if 'invalid_value' in fv:
                invalid_value = fv['invalid_value']
                if type(invalid_value) == str and invalid_value.strip() in ('min', 'max'):
                    if value_type == 'bool':
                        raise ValueError('Field {} is bool type. It should not have min/max as default value')
                    else:
                        (min_value, max_value) = utils.get_min_max(value_type)
                        invalid_value = min_value if invalid_value.strip() == 'min' else max_value
            
            validation_mode = fv.get('validation_mode', 'allow_empty')
            create_flag_field = fv.get('create_flag_field', True) if validation_mode in ('allow_empty', 'relaxed') else False
            flag_field_suffix = fv.get('flag_field_name', '_valid') if create_flag_field else ''

            importer_def = Numeric(value_type, invalid_value, validation_mode, create_flag_field, flag_field_suffix)

        elif field_type == 'datetime':
            create_day_field = fv.get('create_day_field', False)
            create_flag_field = fv.get('create_flag_field', False)
            importer_def = DateTime(create_day_field, create_flag_field)

        elif field_type == 'date':
            create_flag_field = fv.get('create_flag_field', False)
            importer_def = Date(create_flag_field)
            
        else:
            msg = "'{}' is an unsupported field type (For field '{}')."
            raise ValueError(msg.format(field_type, fk))

        schema_dict[fk] = importer_def
        
    return schema_dict





# class DatasetImporter:
#     def __init__(self, datastore, source, hf, space, schema, timestamp,
#                  include=None, exclude=None, keys=None,
#                  stop_after=None, chunk_row_size = (1 << 20)) :
                 

#         if space not in hf.keys():
#             hf.create_group(space)
#         group = hf[space]

#         with open(source, encoding='utf-8') as sf:
#             csvf = csv.DictReader(sf, delimiter=',', quotechar='"')

#             available_keys = [k.strip() for k in csvf.fieldnames if k.strip() in schema.fields]
#             if space in include and len(include[space]) > 0:
#                 available_keys = include[space]
#             if space in exclude and len(exclude[space]) > 0:
#                 available_keys = [k for k in available_keys if k not in exclude[space]]

#             if not keys:
#                 fields_to_use = available_keys
#             else:
#                 for k in keys:
#                     if k not in available_keys:
#                         raise ValueError(f"key '{k}' isn't in the available keys ({keys})")
#                 fields_to_use = keys

#             csvf_fieldnames = [k.strip() for k in csvf.fieldnames]
#             index_map = [csvf_fieldnames.index(k) for k in fields_to_use]


#             field_importer_list = list() # only for field_to_use          
#             for i_n in range(len(fields_to_use)):
#                 field_name = fields_to_use[i_n]
#                 sch = schema.fields[field_name]
#                 field_importer = sch.importer(datastore, group, field_name, timestamp)
#                 field_importer_list.append(field_importer)

#             column_offsets = np.zeros(len(csvf_fieldnames) + 1, dtype=np.int64)
#             for i, field_name in enumerate(csvf_fieldnames):
#                 sch = schema.fields[field_name]
#                 column_offsets[i + 1] = column_offsets[i] + sch.field_size * chunk_row_size
        
#         read_file_using_fast_csv_reader(source, chunk_row_size, column_offsets, index_map, field_importer_list, stop_after_rows=stop_after)


