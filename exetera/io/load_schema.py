from io import StringIO
import json
from typing import Union

from exetera.core.field_importers import Categorical, Numeric, String, DateTime, Date
from exetera.core import validation as val
from exetera.core import utils


def load_schema(source: Union[str, StringIO], verbosity=0):
    schemas = None
    if isinstance(source, str):
        with open(source, encoding='utf-8') as sf:
            schemas = load_schema_file(sf)
    elif isinstance(source, StringIO): 
        schemas = load_schema_file(source)
    return schemas


def load_schema_file(source: Union[str, StringIO], verbosity=0):
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
        schema_dict = schema_file_to_dict(sv)
        spaces[sk] = schema_dict
    return spaces


def schema_file_to_dict(schema):
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
            create_flag_field = fv.get('create_flag_field', False) or fv.get('optional', False)
            importer_def = DateTime(create_day_field, create_flag_field)

        elif field_type == 'date':
            create_day_field = fv.get('create_day_field', False)
            create_flag_field = fv.get('create_flag_field', False) or fv.get('optional', False)
            importer_def = Date(create_day_field, create_flag_field)
            
        else:
            msg = "'{}' is an unsupported field type (For field '{}')."
            raise ValueError(msg.format(field_type, fk))

        schema_dict[fk] = importer_def
        
    return schema_dict