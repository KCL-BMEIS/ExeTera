from collections import defaultdict
import numpy as np
import dataset
from processing.age_from_year_of_birth import CalculateAgeFromYearOfBirth
from utils import valid_range_fac_inc
import utils

def check_age_ranges(pfilename, tfilename, verbosity, territories):

    early_filter =\
        None if territories is None else ('country_code', lambda x: x in territories)

    with open(pfilename) as f:
        ds = dataset.Dataset(f, keys=('id', 'year_of_birth'),
                             early_filter=early_filter,
                             show_progress_every=1000000 if verbosity is 1 else None)
    pids = ds.field_by_name('id')
    pyobs = ds.field_by_name('year_of_birth')
    ages = np.zeros(ds.row_count(), dtype=np.uint32)
    filter = np.zeros(ds.row_count(), dtype=np.uint32)
    fn = CalculateAgeFromYearOfBirth(0x1, 0x2, valid_range_fac_inc(0, 100), 2020)
    fn(pyobs, ages, filter)

    under18 = 0
    eqandover18 = 0
    patientsunder18 = set()
    for i_r in range(ds.row_count()):
        if filter[i_r] == 0:
            if ages[i_r] < 18:
                under18 += 1
                patientsunder18.add(pids[i_r])
            else:
                eqandover18 += 1

    print('under 18:', under18)
    print(len(patientsunder18))
    print('18 and over:', eqandover18)
    print(under18 + eqandover18)
    print(np.count_nonzero(filter == 0))

    del ds

    with open(tfilename) as f:
        ds = dataset.Dataset(f, keys=('patient_id', 'result'),
                             early_filter=early_filter,
                             show_progress_every=100000 if verbosity is 1 else None)

    tpids = ds.field_by_name('patient_id')
    tres = ds.field_by_name('result')
    print(utils.build_histogram(tres))

    class TestResult:
        rmap = {'not_tested': -1, 'waiting': 0, 'failed': 1, 'negative': 2, 'positive': 3}
        def __init__(self):
            self.result = TestResult.rmap['not_tested']
        def add(self, value):
            if TestResult.rmap[value] > self.result:
                self.result = TestResult.rmap[value]


    patient_test_results = defaultdict(TestResult)
    for i_r in range(ds.row_count()):
        if tpids[i_r] in patientsunder18:
            patient_test_results[tpids[i_r]].add(tres[i_r])

    print('under 18 test results:',
          utils.build_histogram([v.result for v in patient_test_results.values()]))

    patient_test_results = defaultdict(TestResult)
    for i_r in range(ds.row_count()):
        patient_test_results[tpids[i_r]].add(tres[i_r])

    print('overall test results:',
          utils.build_histogram([v.result for v in patient_test_results.values()]))


if __name__ == '__main__':

    from standardargs import standard_args

    args = standard_args(use_patients=True, use_tests=True).parse_args()

check_age_ranges(args.patient_data, args.test_data, args.verbosity, args.territories)