import h5py

import dataset
import data_schemas
import utils

pfilename = '/home/ben/covid/assessments_export_20200617030002.csv'
# keys = None
keys = ('worn_face_mask', 'mask_cloth_or_scarf', 'mask_surgical', 'mask_not_sure_pfnts', 'mask_n95_ffp',
        'typical_hayfever', 'mask_other')
data_schema = data_schemas.DataSchema(1)

with open(pfilename) as f:
    ds = dataset.Dataset(f, stop_after=1, show_progress_every=1000000)
names = ds.names_
snames = set(ds.names_)
print(snames.difference(data_schema.assessment_field_types.keys()))
del ds

step = 4
start = 0
end = min(step, len(keys))
while True:
    print('start =', start)
    with open(pfilename) as f:
        ds = dataset.Dataset(f, keys=keys, show_progress_every=1000000)
        for n in keys[start:end]:
            if data_schema.assessment_field_types.get(n, 'categoricaltype') == 'categoricaltype':
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
        end = min(end+step, len(keys))
    del ds
