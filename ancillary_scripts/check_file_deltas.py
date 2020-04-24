import dataset
import pipeline


filename1 = 'assessments_short.csv'
filename2 = 'assessments_short_0423.csv'


with open(filename1) as f:
    ds1 = dataset.Dataset(f, progress=True)

with open(filename2) as f:
    ds2 = dataset.Dataset(f, progress=True)


print(ds1.row_count())
ds1.sort(('patient_id', 'updated_at'))
print(ds2.row_count())
ds2.sort(('patient_id', 'updated_at'))

fields = set(ds1.names_).intersection(set(ds2.names_))
print(fields)

def match_rows(k1, k2):
    x = 0
    y = 0
    xindices = list()
    yindices = list()
    while x < len(k1) and y < len(k2):
        if k1[x] < k2[y]:
            x += 1
        elif k1[x] > k2[y]:
            y += 1
        else:
            xindices.append(x)
            yindices.append(y)
            x += 1
            y += 1
    return xindices, yindices

def elements_not_equal(xinds, yinds, f1, f2):
    discrepencies = None
    for r in range(len(xinds)):
        x = xinds[r]
        y = yinds[r]
        if f1[x] != f2[y]:
            if discrepencies is None:
                discrepencies = list()
            discrepencies.append(r)

    return discrepencies

k1 = ds1.field_by_name('id')
k2 = ds2.field_by_name('id')
xinds, yinds = match_rows(k1, k2)

diagnostic_keys = ['id', 'patient_id', 'created_at', 'updated_at']
for f in fields:

    f1 = ds1.field_by_name(f)
    f2 = ds2.field_by_name(f)
    discrepencies = elements_not_equal(xinds, yinds, f1, f2)
    if discrepencies is not None:
        for d in discrepencies[0:10]:
            v1 = ds1.value_from_fieldname(xinds[d], f)
            v2 = ds2.value_from_fieldname(yinds[d], f)
            print(xinds[d], yinds[d],
                  'na' if v1 == '' else v1, '|', 'na' if v2 == '' else v2)
            pipeline.print_diagnostic_row(xinds[d], ds1, xinds[d], diagnostic_keys + [f])
            pipeline.print_diagnostic_row(yinds[d], ds2, yinds[d], diagnostic_keys + [f])
#
# for r in range(ds1.row_count()):
#     for f in fields:
#         v1 = ds1.value_from_fieldname(r, f)
#         v2 = ds2.value_from_fieldname(r, f)
#         if v1 != v2:
#             print('discrepency:', r, f, v1, v2)
#
# is_carer_for_community = ds1.field_by_name('is_carer_for_community')
# print(is_carer_for_community.count(''))
#
# for n in ds2.names_:
#     if elements_equal(is_carer_for_community, ds2.field_by_name(n)):
#         print('match with ', n)
