import numpy as np

from exetera.core import validation as val


def combined_hcw_with_contact(datastore,
                              healthcare_professional, contact_health_worker,
                              is_carer_for_community,
                              group, name):
    raw_hcp = val.raw_array_from_parameter(datastore, 'healthcare_professional',
                                           healthcare_professional)
    filter_ = np.where(raw_hcp == 0,
                       0,
                       np.where(raw_hcp == 1,
                                1,
                                np.where(raw_hcp < 4,
                                         2,
                                         3)))
    raw_chw = val.raw_array_from_parameter(datastore, 'contact_health_worker',
                                           contact_health_worker)
    filter_ = np.maximum(filter_, np.where(raw_chw == 2, 3, raw_chw))

    raw_icfc = val.raw_array_from_parameter(datastore, 'is_carer_for_community',
                                            is_carer_for_community)
    filter_ = np.maximum(filter_,
                         np.where(raw_icfc == 2, 3, raw_icfc))
    key = {'': 0, 'no': 1, 'yes_no_contact': 2, 'yes_contact': 3}
    hccw = datastore.get_categorical_writer(group, name, categories=key)
    hccw.write(filter_)
    return hccw
