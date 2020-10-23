
from collections import defaultdict

from exetera.core import dataset
from exetera.processing import analytics


class FilterNewAssessmentsV1:

    def __init__(self, test_ds, patients):
        if not isinstance(test_ds, dataset.Dataset):
            raise ValueError("'test_ds' must be of type Dataset")
        if not isinstance(patients, type(defaultdict(analytics.TestIndex))):
            raise ValueError("'patients' must be of type 'defaultdict(TestIndex)'")
        self.test_ds = test_ds
        self.patients = patients

    def __call__(self):
        tids = self.test_ds.field_by_name('id')
        pids = self.test_ds.field_by_name('patient_id')
        cats = self.test_ds.field_by_name('created_at')
        edates = self.test_ds.field_by_name('date_taken_specific')
        edate_los = self.test_ds.field_by_name('date_taken_between_start')
        edate_his = self.test_ds.field_by_name('date_taken_between_end')

        cleaned_patients = defaultdict(analytics.TestIndices)
        for p in self.patients.items():
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

                for e in sorted(cleaned_entries.items(), key=lambda x: x[0]):
                    last_index = e[1][0]
                    cleaned_patients[p[0]].add(last_index)

        return cleaned_patients

class ConvertOldAssessmentsV1:

    result_map = {0: 0, 1: 1, 2: 3, 3: 4}
    value_map = {0: 0, 1: 1, 2: 2, 3: 3}

    def __init__(self, assessment_ds, test_ds, value_map=None):
        self.assessment_ds = assessment_ds
        self.test_ds = test_ds
        self.value_map =\
            ConvertOldAssessmentsV1.value_map if value_map is None else value_map

    def __call__(self, patient_assessment_indices, test_patients, assessment_flags):

        results = {
            'id': list(), 'patient_id': list(), 'created_at': list(), 'updated_at': list(),
            'version': list(), 'country_code': list(), 'result': list(), 'mechanism': list(),
            'date_taken_specific': list(),
            'date_taken_between_start': list(), 'date_taken_between_end': list()
        }
        aids = self.assessment_ds.field_by_name('id')
        pids = self.assessment_ds.field_by_name('patient_id')
        cats = self.assessment_ds.field_by_name('updated_at')
        uats = self.assessment_ds.field_by_name('created_at')
        vrsns = self.assessment_ds.field_by_name('version')
        ccs = self.assessment_ds.field_by_name('country_code')
        tcps = self.assessment_ds.field_by_name('tested_covid_positive')

        for k, v in patient_assessment_indices.items():
            if k not in test_patients:
                # determine whether the record is coherent
                flagged = False
                counts = [0, 0, 0, 0]
                for i in v.indices:
                    if assessment_flags[i]:
                        flagged = True
                        break
                    counts[self.value_map[tcps[i]]] += 1
                if not flagged:
                    first = v.indices[0]
                    if counts[2] > 0 and counts[3] > 0:
                        print(f"no: {counts[2]}, yes: {counts[3]}")
                    else:
                        if counts[3] > 0:
                            result = ConvertOldAssessmentsV1.result_map[3]
                        elif counts[2] > 0:
                            result = ConvertOldAssessmentsV1.result_map[2]
                        elif counts[1] > 0:
                            result = ConvertOldAssessmentsV1.result_map[1]
                        self.generate_test(results,
                                           aids[first], pids[first],
                                           cats[first], uats[first],
                                           vrsns[first], ccs[first], result, '',
                                           cats[first])
        return results

    def generate_test(self, results,
                      id, pid, created_at, updated_at, version, country_code, result,
                      mechanism, date_taken):
        results['id'].append(id)
        results['patient_id'].append(pid)
        results['created_at'].append(created_at)
        results['updated_at'].append(updated_at)
        results['version'].append(version)
        results['country_code'].append(country_code)
        results['result'].append(result)
        results['mechanism'].append(mechanism)
        results['date_taken_specific'].append(date_taken)
        results['date_taken_between_start'].append('')
        results['date_taken_between_end'].append('')



