import sys

from hytable.core import dataset
from hytable.covidspecific import data_schemas

def check_missing_fields(entries, categorical_maps, filename):
    print(filename)
    names = None
    with open(filename) as f:
        ds = dataset.Dataset(f, stop_after=1)
        names = ds.names_

    missing_fields = []
    for n in names:
        if n not in entries:
            missing_fields.append(n)

    print(missing_fields)
    if len(missing_fields) == 0:
        return

    with open(filename) as f:
        ds = dataset.Dataset(f, keys=missing_fields + ['version'], show_progress_every=1000000)

    for k in missing_fields:
        field = ds.field_by_name(k)
        ufield = set(field)
        if len(ufield) < 20:
            print(k, ufield)
        else:
            print(k, "not categorical", list(ufield)[:10])


if __name__ == '__main__':

    for filename in sys.argv[1:]:
        schema = data_schemas.DataSchema(1)
        if 'patient' in filename:
            check_missing_fields(schema.patient_field_types, schema.patient_field_entries, filename)
        elif 'assessment' in filename:
            check_missing_fields(schema.assessment_field_types, schema.assessment_field_entries, filename)
        elif 'covid' in filename:
            check_missing_fields(schema.test_field_types, schema.test_field_entries, filename)
        else:
            raise ValueError("Can't handle this file")