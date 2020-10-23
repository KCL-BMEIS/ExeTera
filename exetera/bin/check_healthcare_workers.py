from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import h5py

from exetera.core import persistence
from exetera.core.utils import build_histogram
from exetera.core import exporter
from exetera.processing.test_type_from_mechanism import test_type_from_mechanism_v1

"""
# make an all encompassing healthcare worker variable 
data.base <- data.base %>%
  mutate(hcw_contact = case_when(
    contact_health_worker == 'TRUE' ~ 'yes_does_interact', 
    is_carer_for_community == 'TRUE' ~ 'yes_does_interact', #based on the way the question is phrased#
    healthcare_professional == 'yes_does_not_treat' ~ 'yes_does_not_interact', 
    healthcare_professional == 'yes_does_treat' ~ 'yes_does_interact',
    healthcare_professional == 'yes_does_not_interact' ~ 'yes_does_not_interact', 
    healthcare_professional == 'yes_does_interact' ~ 'yes_does_interact'))
data.base$hcw_contact <- fct_explicit_na(data.base$hcw_contact, na_level = "no")
data.base$hcw_contact<- factor(data.base$hcw_contact, 
                               levels = c('no', 'yes_does_not_interact', 'yes_does_interact'))
data.base <- data.base %>%
  mutate(hcw_yn = case_when(
    hcw_contact == 'yes_does_interact' ~ 'TRUE',
    TRUE ~ 'FALSE'))
data.base$hcw_yn <- as.logical(data.base$hcw_yn)
"""


def combined_hcw_contact(healthcare_professional, contact_health_worker, is_carer_for_community,
                         healthcare_worker_with_contact):
    filter_ = np.zeros(len(healthcare_professional), dtype='int8')
    filter_ = np.where(healthcare_professional == 0,
                       0,
                       np.where(healthcare_professional == 1,
                                1,
                                np.where(healthcare_professional < 4,
                                         2,
                                         3)))
    filter_ = np.maximum(filter_,
                         np.where(contact_health_worker == 2, 3, contact_health_worker))

    filter_ = np.maximum(filter_,
                         np.where(is_carer_for_community == 2, 3, is_carer_for_community))
    healthcare_worker_with_contact.write(filter_)


def earliest(dates, field):
    earliest = None
    for i_p, p in enumerate(dates):
        if field[i_p] > 0:
            if earliest is None or p < earliest:
                earliest = p
    return datetime.fromtimestamp(earliest)

filename = '/home/ben/covid/ds_20200720_full.hdf5'
filenametemp = '/home/ben/covid/ds_temp.hdf5'
ds = persistence.DataStore()
with h5py.File(filename, 'r') as src:
    with h5py.File(filenametemp, 'w') as tmp:

        s_pat = src['patients']
        cats = ds.get_reader(s_pat['created_at'])[:]
        hcp = ds.get_reader(s_pat['healthcare_professional'])[:]
        print("healthcare_professional")
        print(sorted(build_histogram(hcp)))
        print(earliest(cats, hcp))

        chw = ds.get_reader(s_pat['contact_health_worker'])[:]
        print(sorted(build_histogram(chw)))
        print(earliest(cats, chw))

        ipwc = ds.get_reader(s_pat['interacted_patients_with_covid'])[:]
        iwc = ds.get_reader(s_pat['interacted_with_covid'])[:]
        icfc = ds.get_reader(s_pat['is_carer_for_community'])[:]

        hwc = ds.get_categorical_writer(
            tmp, 'health_worker_combined',
            {'': 0, 'no': 1, 'yes_no_interaction': 2, 'yes_interaction': 3})
        combined_hcw_contact(hcp, chw, icfc, hwc)
        hwc = ds.get_reader(tmp['health_worker_combined'])[:]

        print("healthcare_professional, contact_health_worker, is_carer_for_community")
        categories = defaultdict(int)
        for i in range(len(hcp)):
            categories[(hcp[i], chw[i], icfc[i], hwc[i])] += 1

        sorted_categories = sorted(list(categories.items()))
        for c in sorted_categories:
            print(c)

        print(sorted(build_histogram(hwc)))

        exporter.export_to_csv('/home/ben/covid/health_worker_combined_20200720.csv',
                               ds, [('id', s_pat['id']),
                                    ('healthcare_professional', s_pat['healthcare_professional']),
                                    ('contact_health_worker', s_pat['contact_health_worker']),
                                    ('is_carer_for_community', s_pat['is_carer_for_community']),
                                    ('health_worker_combined', tmp['health_worker_combined'])])


