import csv

def compare(expected, actual):
    with open(expected) as fpe:
        with open(actual) as fpa:
            csvfpe = csv.DictReader(fpe)
            expected_field_names = csvfpe.fieldnames
            e_permutation = sorted([x for x in range(len(expected_field_names))], key=lambda x: expected_field_names[x])
            print(e_permutation)

            csvfpe = csv.reader(fpe)

            csvfpa = csv.DictReader(fpa)
            actual_field_names = csvfpa.fieldnames
            a_permutation = sorted([x for x in range(len(actual_field_names))], key=lambda x: actual_field_names[x])
            print(a_permutation)

            csvfpa = csv.reader(fpa)

            if expected_field_names == actual_field_names:
                print('field names are identical')
            else:
                print(set(expected_field_names).difference(set(actual_field_names)))
                print(set(actual_field_names).difference(set(expected_field_names)))

            print(csvfpe.line_num)

            efields = list()
            for ei, e in enumerate(csvfpe):
                efields.append(e)
            print(ei+1)

            afields = list()
            for ai, a in enumerate(csvfpa):
                afields.append(a)
            print(ai+1)

            disparities = 0
            for i in range(len(efields)):
                if efields[i] != afields[i]:
                    disparities += 1

            if disparities > 0:
                print('rows not equal')
            else:
                print('rows equal')


epfn = 'expected_50k_patients.csv'
apfn = 'test_patients.csv'
eafn = 'expected_50k_assessments.csv'
aafn = 'test_assessments.csv'
compare(epfn, apfn)
compare(eafn, aafn)
