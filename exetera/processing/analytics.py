from collections.__init__ import defaultdict

from exetera.core import utils


class TestIndices:
    def __init__(self):
        self.indices = list()

    def add(self, index):
        self.indices.append(index)

    def __repr__(self):
        return f"{self.indices}"


# TODO: move to a separate python package that is application specific to zoe pipeline
class OldFormatTestSummary:
    def __init__(self):
        self.indices = list()
        self.seen_negative = False
        self.seen_positive = False

    def add(self, index, result):
        self.indices.append(index)
        self.seen_negative = self.seen_negative or result == 'no'
        self.seen_positive = self.seen_positive or result == 'yes'


# TODO: move to a separate python package that is application specific to zoe pipeline
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


# TODO: move to a separate python package that is application specific to zoe pipeline
def get_patients_with_old_format_tests(a_ds):
    apids = a_ds.field_by_name('patient_id')
    ahcts = a_ds.field_by_name('had_covid_test')
    atcps = a_ds.field_by_name('tested_covid_positive')
    auats = a_ds.field_by_name('updated_at')
    print('row count:', a_ds.row_count())

    apatients = defaultdict(OldFormatTestSummary)
    for i_r in range(a_ds.row_count()):
        if ahcts[i_r] == 'True' or atcps[i_r] in ('no', 'yes'):
            apatients[apids[i_r]].add(i_r, atcps[i_r])

    apatient_test_count = 0
    for k, v in apatients.items():
        if len(v.indices) > 0:
            apatient_test_count += 1

    return apatients


# TODO: move to a separate python package that is application specific to zoe pipeline
def filter_duplicate_new_tests(ds, patients, threshold_for_diagnostic_print=1000000):
    tids = ds.field_by_name('id')
    pids = ds.field_by_name('patient_id')
    cats = ds.field_by_name('created_at')
    edates = ds.field_by_name('date_taken_specific')
    edate_los = ds.field_by_name('date_taken_between_start')
    edate_his = ds.field_by_name('date_taken_between_end')

    cleaned_patients = defaultdict(TestIndices)
    for p in patients.items():
        # print(p[0], len(p[1].indices))
        test_dates = set()
        for i_r in reversed(p[1].indices):
            test_dates.add((edates[i_r], edate_los[i_r], edate_his[i_r]))
        if len(test_dates) == 1:
            istart = p[1].indices[-1]
            # utils.print_diagnostic_row(f"{istart}", ds, istart, ds.names_)
            cleaned_patients[p[0]].add(istart)
        else:
            cleaned_entries = dict()
            for t in test_dates:
                cleaned_entries[t] = list()
            for i_r in reversed(p[1].indices):
                cleaned_entries[(edates[i_r], edate_los[i_r], edate_his[i_r])].append(i_r)

            if len(test_dates) > threshold_for_diagnostic_print:
                print(p[0])
            for e in sorted(cleaned_entries.items(), key=lambda x: x[0]):
                last_index = e[1][0]
                if len(test_dates) > threshold_for_diagnostic_print:
                    utils.print_diagnostic_row(f"{p[0]}/{last_index}", ds, last_index, ds.names_)
                cleaned_patients[p[0]].add(last_index)

    return cleaned_patients
