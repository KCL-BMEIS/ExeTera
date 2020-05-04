import csv
import dataset

import split

def jury_rigged_split_sanity_check():
    patient_filename = '/home/ben/covid/patients_export_geocodes_20200504030002.csv'
    assessment_filename = '/home/ben/covid/assessments_export_20200504030002.csv'

    patient_keys = ('id', 'created_at', 'year_of_birth')
    with open(patient_filename) as pds:
        p_ds = dataset.Dataset(pds, keys=patient_keys,
                               progress=True)

    p_ds.sort(keys=('created_at', 'id'))


    small_p_dses = list()
    for i in range(8):
        with open(patient_filename[:-4] + f"_{i:04d}" + ".csv") as spds:
            small_p_dses.append(dataset.Dataset(spds, keys=patient_keys,
                                                progress=True))


    print('p_ds:', p_ds.row_count())
    for i_d in range(len(small_p_dses)):
        print(f'spds{i_d}', small_p_dses[i_d].row_count())


    p_ids = p_ds.field_by_name('id')
    # p_id_indices = dict()
    # for i_p, p in enumerate(p_ids):
    #     p_id_indices[p] = i_p
    p_c_ats = p_ds.field_by_name('created_at')
    p_yobs = p_ds.field_by_name('year_of_birth')

    accumulated = 0
    for d in range(8):
        print('checking subset', d)
        sp_ids = small_p_dses[d].field_by_name('id')
        sp_c_ats = small_p_dses[d].field_by_name('created_at')
        sp_yobs = small_p_dses[d].field_by_name('year_of_birth')
        for i_r in range(small_p_dses[d].row_count()):
            if sp_ids[i_r] != p_ids[accumulated + i_r]:
                print(i_r, 'ids do not match')
            if sp_c_ats[i_r] != p_c_ats[accumulated + i_r]:
                print(i_r, 'updated ats do not match')
            if sp_yobs[i_r] != p_yobs[accumulated + i_r]:
                print(i_r, 'year of births do not match')
        accumulated += small_p_dses[d].row_count()



jury_rigged_split_sanity_check()