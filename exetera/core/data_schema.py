import exetera
from exetera.core import readerwriter as rw


# field_writers = {
#     'idtype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 32, ts),
#     'datetimetype':
#         lambda g, cs, n, ts: rw.DateTimeImporter(g, cs, n, False, ts),
#     'optionaldatetimetype':
#         lambda g, cs, n, ts: rw.DateTimeImporter(g, cs, n, True, ts),
#     'datetype':
#         lambda g, cs, n, ts: rw.OptionalDateImporter(g, cs, n, False, ts),
#     'optionaldatetype':
#         lambda g, cs, n, ts: rw.OptionalDateImporter(g, cs, n, True, ts),
#     'versiontype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 10, ts),
#     'indexedstringtype': lambda g, cs, n, ts: rw.IndexedStringWriter(g, cs, n, ts),
#     'countrycodetype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 2, ts),
#     'unittype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 1, ts),
#     'categoricaltype':
#         lambda g, cs, n, stv, ts: rw.CategoricalWriter(g, cs, n, stv, ts),
#     'leakycategoricaltype':
#         lambda g, cs, n, stv, oor, ts: rw.LeakyCategoricalImporter(g, cs, n, stv,
#                                                                    oor, ts),
#     'float32type': lambda g, cs, n, ts: rw.NumericImporter(
#         g, cs, n, 'float32', pers.try_str_to_float, ts),
#     'uint16type': lambda g, cs, n, ts: rw.NumericImporter(
#         g, cs, n, 'uint16', pers.try_str_to_int, ts),
#     'yeartype': lambda g, cs, n, ts: rw.NumericImporter(
#         g, cs, n, 'uint32', pers.try_str_to_float_to_int, ts),
#     'geocodetype': lambda g, cs, n, ts: rw.FixedStringWriter(g, cs, n, 9, ts)
# }

# new_field_writers = {
#     'idtype': lambda s, g, n, ts=None, cs=None:
#     fields.FixedStringImporter(s, g, n, 32, ts, cs),
#     'datetimetype': lambda s, g, n, ts=None, cs=None:
#     fields.DateTimeImporter(s, g, n, False, True, ts, cs),
#     'optionaldatetimetype': lambda s, g, n, ts=None, cs=None:
#     fields.DateTimeImporter(s, g, n, True, True, ts, cs),
#     'datetype': lambda s, g, n, ts=None, cs=None:
#     fields.DateImporter(s, g, n, False, ts, cs),
#     'optionaldatetype': lambda s, g, n, ts=None, cs=None:
#     fields.DateImporter(s, g, n, True, ts, cs),
#     'versiontype': lambda s, g, n, ts=None, cs=None:
#     fields.FixedStringImporter(s, g, n, 10, ts, cs),
#     'indexedstringtype': lambda s, g, n, ts=None, cs=None:
#     fields.IndexedStringImporter(s, g, n, ts, cs),
#     'countrycodetype': lambda s, g, n, ts=None, cs=None:
#     fields.FixedStringImporter(s, g, n, 2, ts, cs),
#     'unittype': lambda s, g, n, ts=None, cs=None:
#     fields.FixedStringImporter(s, g, n, 1, ts, cs),
#     'categoricaltype': lambda s, g, n, vt, stv, ts=None, cs=None:
#     fields.CategoricalImporter(s, g, n, vt, stv, ts, cs),
#     'leakycategoricaltype': lambda s, g, n, vt, stv, oor, ts=None, cs=None:
#     fields.LeakyCategoricalImporter(s, g, n, vt, stv, oor, ts, cs),
#     'float32type': lambda s, g, n, ts=None, cs=None:
#     fields.NumericImporter(s, g, n, 'float32', pers.try_str_to_float, ts, cs),
#     'uint16type': lambda s, g, n, ts=None, cs=None:
#     fields.NumericImporter(s, g, n, 'uint16', pers.try_str_to_int, ts, cs),
#     'yeartype': lambda s, g, n, ts=None, cs=None:
#     fields.NumericImporter(s, g, n, 'uint32', pers.try_str_to_float_to_int, ts, cs),
#     'geocodetype': lambda s, g, n, ts=None, cs=None:
#     fields.FixedStringImporter(s, g, n, 9, ts, cs)
# }


new_field_importers = {
    'string': lambda:
        lambda ds, g, n, ts: rw.IndexedStringWriter(ds, g, n, ts),
    'fixed_string': lambda strlen:
        lambda ds, g, n, ts: rw.FixedStringWriter(ds, g, n, strlen, ts),
    'datetime': lambda optional:
        lambda ds, g, n, ts: rw.DateTimeImporter(ds, g, n, optional, ts),
    'date': lambda optional:
        lambda ds, g, n, ts: rw.OptionalDateImporter(ds, g, n, optional, ts),
    'numeric': lambda typestr, parser:
        lambda ds, g, n, ts: rw.NumericImporter(ds, g, n, typestr, parser, ts),
    'categorical': lambda stv, oor=None:
        lambda ds, g, n, ts: rw.CategoricalWriter(ds, g, n, stv, ts) if oor is None else
        rw.LeakyCategoricalImporter(ds, g, n, stv, oor, ts)
}


class FieldDesc:
    def __init__(self, field, importer, strings_to_values, values_to_strings, to_datatype,
                 out_of_range_label):
        self.field = field
        self.importer = importer
        self.to_datatype = to_datatype
        self.strings_to_values = strings_to_values
        self.values_to_strings = values_to_strings
        self.out_of_range_label = out_of_range_label

    def __str__(self):
        output = 'FieldDesc(field={}, strings_to_values={}, values_to_strings={})'
        return output.format(self.field, self.strings_to_values, self.values_to_strings)

    def __repr__(self):
        return self.__str__()


class FieldEntry:
    def __init__(self, field_desc, version_from, version_to=None):
        self.field_desc = field_desc
        self.version_from = version_from
        self.version_to = version_to

    def __str__(self):
        output = 'FieldEntry(field_desc={}, version_from={}, version_to={})'
        return output.format(self.field_desc, self.version_from, self.version_to)

    def __repr__(self):
        return self.__str__()
