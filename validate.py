import csv
import pipeline

equivalence_map = {
    'na':('', 'na'),
    '': ('', 'na')
}

def compare_row(eindex, expected, aindex, actual, keys):
    for k in keys:
        evalue = expected.value_from_fieldname(eindex, k)
        avalue = actual.value_from_fieldname(aindex, k)
        if evalue in equivalence_map:
            return avalue in equivalence_map[evalue]
        else:
            return evalue == avalue

def compare(expected, actual):
    # with open(expected) as fpe:
    #     with open(actual) as fpa:
    #         csvfpe = csv.DictReader(fpe)
    #         expected_field_names = csvfpe.fieldnames
    #         e_permutation = sorted([x for x in range(len(expected_field_names))], key=lambda x: expected_field_names[x])
    #         print(e_permutation)
    #
    #         csvfpe = csv.reader(fpe)
    #
    #         csvfpa = csv.DictReader(fpa)
    #         actual_field_names = csvfpa.fieldnames
    #         a_permutation = sorted([x for x in range(len(actual_field_names))], key=lambda x: actual_field_names[x])
    #         print(a_permutation)
    #
    #         csvfpa = csv.reader(fpa)

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

    # efields = list()
    # for ei, e in enumerate(csvfpe):
    #     efields.append(e)
    # print(ei+1)
    efields = expected_ds.parse_file()
    print(expected_ds.row_count())

    # afields = list()
    # for ai, a in enumerate(csvfpa):
    #     afields.append(a)
    # print(ai+1)
    afields = actual_ds.parse_file()
    print(actual_ds.row_count())

    disparities = 0
    for i in range(len(efields)):
        for n in common_fields:
            compare_row()
        if efields[1][i] != afields[1][i] and efields[1][i] not in ('na', '') and afields[1][i] not in ('na', ''):
            disparities += 1

    if disparities > 0:
        print('rows not equal')
    else:
        print('rows equal')

def validate():
    epfn = 'expected_50k_patients.csv'
    apfn = 'test_patients.csv'
    eafn = 'expected_50k_assessments.csv'
    aafn = 'test_assessments.csv'
    compare(epfn, apfn)
    compare(eafn, aafn)

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

# print()
# print('input')
# print('-----')
# filename = '/home/ben/covid/assessments_short.csv'
# show_rows(filename, ('id', 'patient_id', 'updated_at', 'fatigue'))
#
# print()
# print('python output')
# print('-------------')
# filename = '/home/ben/git/zoe-data-prep/test_assessments.csv'
# show_rows(filename, ('id', 'patient_id', 'updated_at', 'fatigue', 'fatigue_binary'))
#
# print()
# print('R output')
# print('--------')
# filename = '/home/ben/covid/assessments_cleaned_short.csv'
# show_rows(filename, ('id', 'patient_id', 'updated_at', 'fatigue', 'fatigue_binary'))

validate()