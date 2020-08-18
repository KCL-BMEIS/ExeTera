import numpy as np
import pandas as pd
import h5py
from hystore.core.persistence import DataStore
from hystore.core import utils

class DateCounts:
    def __init__(self):
        self.date = 0
        self.multi = False

    def set_date(self, date):
        if date != np.float64(0.0):
            if self.date == np.float64(0.0):
                self.date = date
            elif self.date != date:
                self.date = date
                self.multi = True

from collections import defaultdict
from datetime import datetime

with h5py.File('/home/ben/covid/ds_20200702_full.hdf5', 'r') as hf:
    ds = DataStore()

    print(hf.keys())
    asmts = hf['assessments']
    f_dtog = ds.get_reader(asmts['date_test_occurred_guess'])[:] != 0
    print(np.count_nonzero(f_dtog == True))

    # with utils.Timer("spans"):
    #     spans = ds.get_spans(ds.get_reader(asmts['patient_id'])[:])
    #
    # with utils.Timer("get_max_inds"):
    #     max_inds = ds.apply_spans_index_of_max(spans, ds.get_reader(asmts['had_covid_test'])[:])

    with utils.Timer("checking default dates"):
        pid_dates = defaultdict(DateCounts)

        pids = ds.get_reader(asmts['patient_id'])[:]
        cats = ds.get_reader(asmts['created_at'])[:]
        dcats = ds.get_reader(asmts['date_test_occurred'])[:]
        tcps = ds.get_reader(asmts['tested_covid_positive'])[:]
        locs = ds.get_reader(asmts['location'])[:]
        loc_keys = ds.get_reader(asmts['location']).keys
        print(loc_keys)
        pid_filt = pids == b'000558c92e4f8dc9886d66d580e33331'

        fpids = pids[pid_filt]
        fcats = cats[pid_filt]
        fdcats = dcats[pid_filt]
        ftcps = tcps[pid_filt]
        flocs = locs[pid_filt]
        for i in range(len(fpids)):
            print(fpids[i], datetime.fromtimestamp(fcats[i]),
                  datetime.fromtimestamp(fdcats[i]), ftcps[i], flocs[i])

            # pid_dates[pids[i]].set_date(dcats[i])

        # multi_count = 0
        # for k,v in pid_dates.items():
        #     if v.multi:
        #         multi_count += 1

        #print(multi_count)