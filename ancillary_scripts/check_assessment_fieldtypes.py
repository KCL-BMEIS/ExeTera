import h5py

import dataset
import data_schemas
import utils

pfilename = '/home/ben/covid/assessments_export_20200601030001.csv'
# keys = None
keys = ('date_test_occurred', 'date_test_occurred_guess')
data_schema = data_schemas.DataSchema(1)
with open(pfilename) as f:
    ds = dataset.Dataset(f, keys=keys, stop_after=1, show_progress_every=1000000)
names = ds.names_
del ds

step = 4
start = 0
end = step
while True:
    print('start =', start)
    with open(pfilename) as f:
        ds = dataset.Dataset(f, keys=keys, show_progress_every=1000000)
        for n in names[start:end]:
            if data_schema.assessment_field_types.get(n, None) == 'categoricaltype':
                print(n)
                h = utils.build_histogram(ds.field_by_name(n))
                if len(h) > 100:
                    print('not categorical!', h[:10])
                else:
                    print(h)
            else:
                print(n)
                field = ds.field_by_name(n)
                lengths = list()
                slengths = set()
                for i in range(ds.row_count()):
                    if len(field[i]) not in slengths:
                        print(field[i])
                        slengths.add(len(field[i]))
                    lengths.append(len(field[i]))
                print(utils.build_histogram(lengths))
        if end == len(names):
            break
        start = end
        end = min(end+step, len(names))
    del ds
