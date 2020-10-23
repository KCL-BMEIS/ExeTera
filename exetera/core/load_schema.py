import copy
import json

from exetera.core import data_schema
from exetera.core import persistence as per


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
            elif field_type == 'string':
                importer = data_schema.new_field_importers[field_type]()

            elif field_type == 'fixed_string':
                NewDataSchema._require_key(fk, 'length', fv)
                length = int(fv['length'])
                importer = data_schema.new_field_importers[field_type](length)

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
                else:
                    if value_type not in permitted_numeric_types:
                        msg = "Field {} has an invalid value_type '{}'. Permitted types are {}"
                        raise ValueError(msg.format(fk, value_type, permitted_numeric_types))
                    if value_type in ('float', 'float32', 'float64'):
                        converter = per.try_str_to_float
                    else:
                        converter = per.try_str_to_int

                importer = data_schema.new_field_importers[field_type](value_type, converter)

            elif field_type in ('datetime', 'date'):
                optional = fv.get('optional', False)
                importer = data_schema.new_field_importers[field_type](optional)
            else:
                msg = "'{}' is an unsupported field type (For field '{}')."
                raise ValueError(msg.format(field_type, fk))

            fd = data_schema.FieldDesc(fk, importer, strs_to_vals, vals_to_strs, value_type,
                                       out_of_range_label)

            #fe = data_schema.FieldEntry(fd, 1, None)
            entries[fk] = fd

        return entries


def load_schema(source, verbosity=0):
    d = json.load(source)
    if verbosity > 1:
        print(d.keys())
    fields = d['schema']
    spaces = dict()
    for fk, fv in fields.items():
        nds = NewDataSchema(fk, fv)
        spaces[fk] = nds
    return spaces
