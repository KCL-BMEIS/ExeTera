class FieldDesc:
    def __init__(self, field, strings_to_values, values_to_strings, to_datatype,
                 out_of_range_label):
        self.field = field
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
