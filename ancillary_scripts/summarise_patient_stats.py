import dataset


p_file_name = '/home/ben/covid/patients_export_geocodes_20200513030002.csv'


with open(p_file_name) as f:
    p_ds = dataset.Dataset(f, 'id')
