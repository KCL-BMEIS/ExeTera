import copy
import json

import hystore.core.data_schema as data_schema

class NewDataSchema:
    def __init__(self, name, schema_dict):
        self.name_ = name
        print(name)
        primary_keys = schema_dict.get('primary_keys', None)
        foreign_keys = schema_dict.get('foreign_keys', None)
        fields = schema_dict.get('fields', None)
        self._field_entries = self._build_fields(fields)
        print(self._field_entries)


    @property
    def name(self):
        return self.name_

    def fields(self):
        copy.deepcopy(self._field_entries)

    def categorical_maps(self):
        pass

    @staticmethod
    def _invert_dictionary(value_list):
        inverse = dict()
        for ir, r in enumerate(value_list):
            inverse[r] = ir
        return inverse

    @staticmethod
    def _require_key(context, key, dictionary):
        if key not in dictionary:
            msg = "'{}': '{}' missing from fields".format(context, key)
            raise ValueError(msg)

    @staticmethod
    def _build_fields(fields):
        entries = dict()
        for fk, fv in fields.items():
            print("  {}: {}".format(fk, fv))
            NewDataSchema._require_key(fk, 'field_type', fv)
            field_type = fv['field_type']
            strs_to_vals = None
            vals_to_strs = None
            out_of_range_label = None
            value_type = None
            if field_type in ('categoricaltype', 'leakycategoricaltype'):
                NewDataSchema._require_key(fk, 'categorical', fv)
                categorical = fv['categorical']
                NewDataSchema._require_key(fk, 'strings_to_values', categorical)
                strs_to_vals = categorical['strings_to_values']
                vals_to_strs = NewDataSchema._invert_dictionary(strs_to_vals)
                if field_type == 'leakycategoricaltype':
                    NewDataSchema._require_key(fk, 'out_of_range', categorical)
                    out_of_range_label = categorical['out_of_range']
                NewDataSchema._require_key(fk, 'value_type', categorical)
            fd = data_schema.FieldDesc(fk, strs_to_vals, vals_to_strs, value_type,
                                       out_of_range_label)
            fe = data_schema.FieldEntry(fd, 1, None)
            entries[fk] = fe

        return entries


def load_schema(source):
    d = json.loads(source)
    print(d.keys())
    fields = d['schema']
    for fk, fv in fields.items():
        nds = NewDataSchema(fk, fv)


filename = '/home/ben/covid/covid_schema.json'
with open(filename) as f:
    load_schema(f.read())
