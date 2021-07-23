import copy
import json

from exetera.core import data_schema
from exetera.core import persistence as per
from exetera.core import utils

class NewDataSchema:
    def __init__(self, name, schema_dict, verbosity=0):
        self.name_ = name
        self.verbosity_ = verbosity
        if verbosity > 1:
            print(name)
        primary_keys = schema_dict.get('primary_keys', None)
        foreign_keys = schema_dict.get('foreign_keys', None)
        fields = schema_dict.get('fields', None)
        self.permitted_numeric_types = ('float32', 'float64', 'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32',
                                        'uint32', 'int64')
        self._field_entries = self._build_fields(fields, self.permitted_numeric_types)
        if verbosity > 1:
            print(self._field_entries)


    @property
    def name(self):
        return self.name_

    @property
    def fields(self):
        return copy.deepcopy(self._field_entries)

    def categorical_maps(self):
        pass

    @staticmethod
    def _invert_dictionary(src_dict):
        inverse = dict()
        for k, v in src_dict.items():
            inverse[v] = k
        return inverse

    @staticmethod
    def _require_key(context, key, dictionary):
        if key not in dictionary:
            msg = "'{}': '{}' missing from fields".format(context, key)
            raise ValueError(msg)

    @staticmethod
    def _build_fields(fields, permitted_numeric_types, verbosity=0):
        entries = dict()
        for fk, fv in fields.items():
            if verbosity > 1:
                print("  {}: {}".format(fk, fv))
            NewDataSchema._require_key(fk, 'field_type', fv)
            field_type = fv['field_type']
            strs_to_vals = None
            vals_to_strs = None
            out_of_range_label = None
            value_type = None
            field_size = 0

            if field_type == 'categorical':
                NewDataSchema._require_key(fk, 'categorical', fv)
                categorical = fv['categorical']
                NewDataSchema._require_key(fk, 'strings_to_values', categorical)
                strs_to_vals = categorical['strings_to_values']
                vals_to_strs = NewDataSchema._invert_dictionary(strs_to_vals)
                if 'out_of_range' in categorical:
                    out_of_range_label = categorical['out_of_range']
                NewDataSchema._require_key(fk, 'value_type', categorical)
                importer = data_schema.new_field_importers[field_type](strs_to_vals,
                                                                       out_of_range_label)
                field_size = max([len(k) for k in strs_to_vals])

            elif field_type == 'string':
                importer = data_schema.new_field_importers[field_type]()
                field_size = 10 # guessing

            elif field_type == 'fixed_string':
                NewDataSchema._require_key(fk, 'length', fv)
                length = int(fv['length'])
                importer = data_schema.new_field_importers[field_type](length)
                field_size = length

            elif field_type == 'numeric':
                NewDataSchema._require_key(fk, 'value_type', fv)
                value_type = fv['value_type']
                if 'raw_type' in fv:
                    raw_type = fv['raw_type']
                    if raw_type != 'float32' and value_type != 'int32':
                        msg = ("{}: if raw_type is specified the conversion must be float32 "
                               " to int32 but is {} and {}, respectively")
                        raise ValueError(msg.format(fk, raw_type, value_type))
                    converter = per.try_str_to_float_to_int
                    field_size = 30
                else:
                    if value_type not in permitted_numeric_types:
                        msg = "Field {} has an invalid value_type '{}'. Permitted types are {}"
                        raise ValueError(msg.format(fk, value_type, permitted_numeric_types))
                    if value_type in ('float', 'float32', 'float64'):
                        converter = per.try_str_to_float
                        field_size = 30
                    elif value_type == 'bool':
                        converter = per.try_str_to_bool
                        field_size = 5
                    elif value_type in ('int', 'int8', 'int16', 'int32', 'int64',
                                        'uint8', 'uint16', 'uint32', 'uint64'):
                        converter = per.try_str_to_int
                        field_size = 20
                    else:
                        msg = "Unrecognised value_type '{}' in field '{}'"
                        raise ValueError(msg.format(value_type, fk))
            
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

                importer = data_schema.new_field_importers[field_type](value_type, converter, invalid_value, validation_mode, create_flag_field, flag_field_suffix)

            elif field_type in ('datetime', 'date'):
                create_day_field = fv.get('create_day_field', False)
                optional = fv.get('optional', False)
                importer = data_schema.new_field_importers[field_type](create_day_field, optional)
                # datettime: 32, date:10
                field_size = 32 if field_type == 'datetime' else 10
            else:
                msg = "'{}' is an unsupported field type (For field '{}')."
                raise ValueError(msg.format(field_type, fk))

            fd = data_schema.FieldDesc(fk, importer, strs_to_vals, vals_to_strs, value_type,
                                       out_of_range_label, field_size)

            #fe = data_schema.FieldEntry(fd, 1, None)
            entries[fk] = fd

        return entries


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

    fields = d['schema']
    spaces = dict()
    for fk, fv in fields.items():
        nds = NewDataSchema(fk, fv)
        spaces[fk] = nds
    return spaces
