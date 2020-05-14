from collections.__init__ import defaultdict


class TestIndices:
    def __init__(self):
        self.indices = list()

    def add(self, index):
        self.indices.append(index)

    def __repr__(self):
        return f"{self.indices}"


def group_new_test_indices_by_patient(ds):
    tids = ds.field_by_name('id')
    pids = ds.field_by_name('patient_id')
    cats = ds.field_by_name('created_at')
    edates = ds.field_by_name('date_taken_specific')
    edate_los = ds.field_by_name('date_taken_between_start')
    edate_his = ds.field_by_name('date_taken_between_end')
    print('row count:', ds.row_count())

    patients = defaultdict(TestIndices)
    for i_r in range(ds.row_count()):
        patients[pids[i_r]].add(i_r)
    return patients


def get_patients_with_old_format_tests(a_ds):
    apids = a_ds.field_by_name('patient_id')
    ahcts = a_ds.field_by_name('had_covid_test')
    atcps = a_ds.field_by_name('tested_covid_positive')
    auats = a_ds.field_by_name('updated_at')
    print('row count:', a_ds.row_count())

    apatients = defaultdict(int)
    for i_r in range(a_ds.row_count()):
        if ahcts[i_r] == 'True' or atcps[i_r] in ('waiting', 'no', 'yes'):
            apatients[apids[i_r]] += 1

    apatient_test_count = 0
    for k, v in apatients.items():
        if v > 0:
            apatient_test_count += 1

    return apatients