import numpy as np

def consolidate_test_dates_v1(ds, created_at, date_taken_specific,
                              date_taken_between_start, date_taken_between_end,
                              effective_test_date,
                              initial_filter=None, first_timestamp=None, last_timestamp=None):
    raw_t_cats = created_at[:]
    raw_t_dts = date_taken_specific[:]
    raw_t_dsbs = date_taken_between_start[:]
    raw_t_dsbe = date_taken_between_end[:]

    if initial_filter:
        test_filter = initial_filter[:]

    # remove tests where no dates are set
    cur_filter = np.logical_not((raw_t_dts == 0) & (raw_t_dsbs == 0) & (raw_t_dsbe == 0))
    test_filter = test_filter & cur_filter
    print("standard test filter 1:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where all three dates are set
    cur_filter = np.logical_not((raw_t_dts != 0) & (raw_t_dsbs != 0) & (raw_t_dsbe != 0))
    test_filter = test_filter & cur_filter
    print("standard test filter 2:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where only one of the date range tests is set
    cur_filter = np.logical_not((raw_t_dsbs != 0) & (raw_t_dsbe == 0) |
                                (raw_t_dsbs == 0) & (raw_t_dsbe != 0))
    test_filter = test_filter & cur_filter
    print("standard test filter 3:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where specific date is set but out of range
    cur_filter = \
        (raw_t_dts == 0) | ((raw_t_dts >= first_timestamp) & (raw_t_dts <= last_timestamp))
    test_filter = test_filter & cur_filter
    print("standard test filter 4:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where beginning date is set but out of range
    cur_filter = \
        (raw_t_dsbs == 0) | ((raw_t_dsbs >= first_timestamp) & (raw_t_dsbs <= last_timestamp))
    test_filter = test_filter & cur_filter
    print("standard test filter 5:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where ending date is set but out of range
    cur_filter = \
        (raw_t_dsbe == 0) | ((raw_t_dsbe >= first_timestamp) & (raw_t_dsbe <= last_timestamp))
    test_filter = test_filter & cur_filter
    print("standard test filter 6:", np.count_nonzero(test_filter), len(test_filter))

    test_timestamps = np.where(raw_t_dts != 0,
                               raw_t_dts,
                               raw_t_dsbs + (raw_t_dsbe - raw_t_dsbs) / 2)

    # remove tests where the test date is after the created at date
    cur_filter = test_timestamps <= raw_t_cats
    test_filter = test_filter & cur_filter
    print("standard test filter 7:", np.count_nonzero(test_filter), len(test_filter))

    # t_rsts = ds.get_reader(s_tests['result'])
    # t_rsts.get_writer(flat_tests, 'result').write(ds.apply_filter(test_filter, t_rsts))
    # t_pids = ds.get_reader(s_tests['patient_id'])
    # t_pids.get_writer(flat_tests, 'patient_id').write(ds.apply_filter(test_filter, t_pids))
    effective_test_date.write(ds.apply_filter(test_filter, test_timestamps))
