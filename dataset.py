import csv
import numpy as np


class Dataset:

    def __init__(self, source):
        self.names_ = list()
        self.fields_ = list()
        self.names_ = list()
        self.index_ = None

        if source:
            csvf = csv.DictReader(source, delimiter=',', quotechar='"')
            self.names_ = csvf.fieldnames

        newline_at = 10
        lines_per_dot = 100000
        # TODO: better for the Dataset to take a stream rather than a name - this allows us to unittest it from strings
        csvf = csv.reader(source, delimiter=',', quotechar='"')

        ecsvf = iter(csvf)
        # next(ecsvf)
        for i, fields in enumerate(ecsvf):
            self.fields_.append(fields)
        #     if i > 0 and i % lines_per_dot == 0:
        #         if i % (lines_per_dot * newline_at) == 0:
        #             print(f'. {i}')
        #         else:
        #             print('.', end='')
        # if i % (lines_per_dot * newline_at) != 0:
        #     print(f' {i}')
        self.index_ = np.asarray([i for i in range(len(self.fields_))], dtype=np.uint32)
        # return strings

    def sort(self, keys):
        #map names to indices
        kindices = [self.field_to_index(k) for k in keys]

        def index_sort(indices):
            def inner_(r):
                return tuple(self.fields_[r][i] for i in indices)
            return inner_

        self.index_ = sorted(self.index_, key=index_sort(kindices))
        fields2 = self.fields_[:]
        Dataset._apply_permutation(self.index_, self.fields_)

    @staticmethod
    def _apply_permutation(permutation, fields):
        n = len(permutation)
        for i in range(0, n):
            pi = permutation[i]
            while pi < i:
                pi = permutation[pi]
            fields[i], fields[pi] = fields[pi], fields[i]
        return fields

    def field_to_index(self, field_name):
        return self.names_.index(field_name)

    def value(self, row_index, field_index):
        return self.fields_[row_index][field_index]

    def value_from_fieldname(self, index, field_name):
        return self.fields_[index][self.field_to_index(field_name)]

    def row_count(self):
        return len(self.fields_)

    def show(self):
        for ir, r in enumerate(self.names_):
            print(f'{ir}-{r}')
