from collections import defaultdict
import numpy as np

import analytics
import dataset
from processing.age_from_year_of_birth import CalculateAgeFromYearOfBirth
import utils

def check_active_gender_and_age(pfilename, afilename, verbosity, territories):
    pkeys = ('id', 'gender', 'year_of_birth')
    akeys = ('patient_id', 'created_at')

    early_filter =\
        None if territories is None else ('country_code', lambda x: x in territories)

    with open(pfilename) as f:
        pds = dataset.Dataset(f, keys=pkeys, early_filter=early_filter,
                              show_progress_every=1000000 if verbosity is 1 else None)
    pids = pds.field_by_name('id')
    pgdrs = pds.field_by_name('gender')
    print(utils.build_histogram(pgdrs))
    pyobs = pds.field_by_name('year_of_birth')

    pfilter = np.zeros(pds.row_count(), dtype=np.uint32)
    ages = np.zeros(pds.row_count(), dtype=np.uint32)
    age_fn = CalculateAgeFromYearOfBirth(0x1, 0x2, utils.valid_range_fac_inc(39, 90), 2020)
    age_fn(pyobs, ages, pfilter)

    patients = set()
    for i_r in range(pds.row_count()):
        if pfilter[i_r] == 0x0 and ages[i_r] >= 40 and pgdrs[i_r] == '1':
            patients.add(pids[i_r])
    del pds

    print(len(patients))

    passessments = defaultdict(analytics.TestIndices)

    with open(afilename) as f:
        ads = dataset.Dataset(f, keys=akeys, early_filter=early_filter,
                              show_progress_every=1000000 if verbosity is 1 else None)
    apids = ads.field_by_name('patient_id')
    acats = ads.field_by_name('created_at')
    for i_r in range(ads.row_count()):
        if apids[i_r] in patients:
            passessments[apids[i_r]].add(i_r)

    passessmentcounts = defaultdict(int)
    for pk, pv in passessments.items():
        sindices = sorted(pv.indices, key=lambda i: acats[i])
        passessmentcounts[pk] = len(sindices)
    print(sorted(utils.build_histogram(passessmentcounts.values())))


if __name__ == '__main__':

    from standardargs import standard_args
    args = standard_args(use_patients=True, use_assessments=True).parse_args()

check_active_gender_and_age(args.patient_data, args.assessment_data, args.verbosity,
                            args.territories)