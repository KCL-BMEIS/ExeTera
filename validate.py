import csv
import pipeline

equivalence_map = {
    'na': ('', 'na'),
    '': ('', 'na')
}

def compare_row(eindex, expected, aindex, actual, keys):
    all_true = True
    for k in keys:
        evalue = expected.value_from_fieldname(eindex, k)
        avalue = actual.value_from_fieldname(aindex, k)
        if evalue in equivalence_map:
            all_true = all_true and avalue in equivalence_map[evalue]
        else:
            all_true = all_true and evalue == avalue

    return all_true

def compare(expected, actual):

    expected_ds = pipeline.Dataset(expected)
    expected_field_names = expected_ds.names_
    actual_ds = pipeline.Dataset(actual)
    actual_field_names = actual_ds.names_

    if expected_field_names == actual_field_names:
        print('field names are identical')
    else:
        print('only in expected:', set(expected_field_names).difference(set(actual_field_names)))
        print('only in actual:', set(actual_field_names).difference(set(expected_field_names)))

    # print(csvfpe.line_num)

    common_fields = set(expected_field_names).intersection(set(actual_field_names))
    print('common fields:', common_fields)

    expected_ds.parse_file()
    print(expected_ds.row_count())

    actual_ds.parse_file()
    print(actual_ds.row_count())

    disparities = 0
    for i in range(expected_ds.row_count()):
            if not compare_row(i, expected_ds, i, actual_ds, common_fields):
                disparities += 1

    if disparities > 0:
        print('rows not equal')
    else:
        print('rows equal')

def validate(expected_patients_file_name, actual_patient_file_name,
             expected_assessments_file_name, actual_assessments_file_name):
    compare(expected_patients_file_name, actual_patient_file_name)
    compare(expected_assessments_file_name, actual_assessments_file_name)

def validate_full():
    filename = '/home/ben/covid/assessments_20200413050002_clean.csv'


    with open(filename) as asmt:
        csvr = csv.DictReader(asmt)
        fieldnames = csvr.fieldnames
        print(fieldnames)

        for ir, r in enumerate(csvr):
            if r['tested_covid_positive'] != '':
                print(ir, r['id'], r['patient_id'], r['tested_covid_positive'])

def show_rows(filename, fields_to_show):
    ds = pipeline.Dataset(filename)
    ds.parse_file()
    ds.sort(('patient_id', 'updated_at'))
    for i_f, f in enumerate(ds.fields_):
        if ds.value_from_fieldname(i_f, 'patient_id') == 'e88bfabe5b16897866f91deb8a7a90f2':
            pipeline.print_diagnostic_row(f'{i_f}:', ds, ds.fields_, i_f, fields_to_show)


epfn = 'v0.1.2_50k_patients.csv'
apfn = 'test_patients.csv'
eafn = 'v0.1.2_50k_assessments.csv'
aafn = 'test_assessments.csv'
validate(epfn, apfn, eafn, aafn)