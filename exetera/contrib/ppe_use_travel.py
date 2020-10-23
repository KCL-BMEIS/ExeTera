import math
from datetime import datetime
from collections import defaultdict
import numpy as np
import h5py
import pandas as pd

from exetera.core import exporter, persistence, utils, session, fields
from exetera.core.persistence import DataStore
from exetera.processing.nat_medicine_model import nature_medicine_model_1
from exetera.processing.method_paper_model import method_paper_model

"""
Goal
 * show whether mask wearing helps with exposure
   * must separate out preventative / remedial action
     * is the person sick and if so, when
     * has the person been predicted as having covid and if so, when
     * has the person had a positive covid test result and if so, when
   * preventative
     * healthy
     * not healthy but not yet covid
   * remedial
     * having had covid
   * must separate out confounding factors
     * background risk
       * non-health worker / non-covid health worker / covid health worker

Process
1. generate results for tested people
  1. filter out people without tests
  
n. generate results for predicted people
"""

def ppe_use_and_travel(ds, src, tmp, start_timestamp):

    logging = True

    s_asmts = src['assessments']

    if 'filtered_assessments' not in tmp.keys():
        f_asmts = tmp.create_group('filtered_assessments')
        cats = ds.get_reader(s_asmts['created_at'])
        asmt_filter = cats[:] >= start_timestamp

        ccs = ds.get_reader(s_asmts['country_code'])
        asmt_filter = asmt_filter & (ccs[:] == b'GB')

        symptom_keys = ('persistent_cough', 'fatigue', 'delirium', 'shortness_of_breath',
                        'fever', 'diarrhoea', 'abdominal_pain', 'chest_pain', 'hoarse_voice',
                        'skipped_meals', 'loss_of_smell')
        mask_keys = ('mask_cloth_or_scarf', 'mask_surgical', 'mask_n95_ffp')
        isolation_keys = ('isolation_healthcare_provider', 'isolation_little_interaction',
                          'isolation_lots_of_people')
        other_keys = ('patient_id', )
        symptom_thresholds = {s: 2 for s in symptom_keys}
        symptom_thresholds.update({m: 2 for m in mask_keys})
        symptom_thresholds['fatigue'] = 3
        symptom_thresholds['shortness_of_breath'] = 3

        for k in symptom_keys + mask_keys + isolation_keys + other_keys:
            with utils.Timer("filtering {}".format(k)):
                reader = ds.get_reader(s_asmts[k])
                if k in mask_keys + symptom_keys:
                    values = np.where(reader[:] >= symptom_thresholds[k], 1, 0)
                    ds.get_numeric_writer(f_asmts, k, 'int8').write(
                        ds.apply_filter(asmt_filter, values))
                    hist = np.unique(reader[:], return_counts=True)
                    print(sorted(zip(hist[0], hist[1])))
                    hist = np.unique(values, return_counts=True)
                    print(sorted(zip(hist[0], hist[1])))
                else:
                    reader.get_writer(f_asmts, k).write(ds.apply_filter(asmt_filter, reader))

        print('filtered assessments:', np.count_nonzero(asmt_filter), len(asmt_filter))
    #
    #
    # if 'filtered_assessment_predictions' not in tmp.keys():
    #     f_pred_asmts = tmp.create_group('filtered_assessment_predictions')
        symptom_readers = dict()
        for s in symptom_keys:
            symptom_readers[s] = ds.get_reader(f_asmts[s])
        predictions = ds.get_numeric_writer(f_asmts, 'prediction', 'float32')
        method_paper_model(ds, symptom_readers, predictions)
        predictions = ds.get_reader(f_asmts['prediction'])
        print('predictions:', np.count_nonzero(predictions[:] > 0), len(predictions))


    if 'patient_assessment_summaries' not in tmp.keys():
        asmt_psum = tmp.create_group('patient_assessment_summaries')
        pids = ds.get_reader(f_asmts['patient_id'])
        mcos = ds.get_reader(f_asmts['mask_cloth_or_scarf'])
        msurg = ds.get_reader(f_asmts['mask_surgical'])
        m95 = ds.get_reader(f_asmts['mask_n95_ffp'])
        with utils.Timer("generating patient_id spans"):
            asmt_spans = ds.get_spans(field=pids[:])

        for k in mask_keys:
            with utils.Timer("getting per patient mask summary for {}".format(k)):
                writer = ds.get_numeric_writer(asmt_psum, k, 'int8')
                ds.apply_spans_max(asmt_spans, ds.get_reader(f_asmts[k])[:], writer)
                print(sorted(utils.build_histogram(ds.get_reader(asmt_psum[k])[:])))

        for k in isolation_keys:
            with utils.Timer("getting per patient isolation summary for {}".format(k)):
                writer = ds.get_numeric_writer(asmt_psum, k, 'int32')
                ds.apply_spans_max(asmt_spans, ds.get_reader(f_asmts[k])[:], writer)
                print(sorted(utils.build_histogram(ds.get_reader(asmt_psum[k])[:])))

        with utils.Timer("getting prediction maxes for patients"):
            p_predictions = predictions.get_writer(asmt_psum, 'prediction')
            ds.apply_spans_max(asmt_spans, predictions, p_predictions)
            p_predictions = ds.get_reader(asmt_psum[k])
            positives = p_predictions[:] > 0
            print("max covid prediction:", np.count_nonzero(positives), len(positives))

        with utils.Timer("getting patient ids from assessments"):
            writer = pids.get_writer(asmt_psum, 'patient_id')
            writer.write(pd.unique(pids[:]))
    else:
        asmt_psum = tmp['patient_assessment_summaries']


    s_ptnts = src['patients']
    print(s_ptnts.keys())

    pdf = pd.DataFrame({'id': ds.get_reader(s_ptnts['id'])[:],
                        'hwwc': ds.get_reader(s_ptnts['health_worker_with_contact'])[:]})
    adf = pd.DataFrame({'patient_id': ds.get_reader(asmt_psum['patient_id'])[:]})
    jdf = pd.merge(left=adf, right=pdf, left_on='patient_id', right_on='id', how='left')
    print(len(jdf['hwwc']))
    class TestResults:
        def __init__(self):
            self.positive = 0
            self.total = 0

        def add(self, result):
            if result:
                self.positive += 1
            self.total += 1

    results = defaultdict(TestResults)
    positives = ds.get_reader(asmt_psum['prediction'])[:]
    positives = positives > 0
    mask_0 = ds.get_reader(asmt_psum['mask_cloth_or_scarf'])[:]
    mask_1 = ds.get_reader(asmt_psum['mask_surgical'])[:]
    mask_2 = ds.get_reader(asmt_psum['mask_cloth_or_scarf'])[:]
    # mask = mask_0 | mask_1 | mask_2
    mask = mask_0
    print(np.unique(mask, return_counts=True))
    isol_lots = ds.get_reader(asmt_psum['isolation_lots_of_people'])[:]
    isol_lots_7 = np.where(isol_lots > 7, 7, isol_lots)
    print(np.unique(isol_lots_7, return_counts=True))
    print(len(mask), len(positives), len(isol_lots_7))

    # isolation lots of users
    for i_r in range(len(mask)):
        results[(isol_lots_7[i_r], mask[i_r])].add(positives[i_r])

    groupings = sorted(list((r[0], (r[1].positive, r[1].total)) for r in results.items()))

    for g in groupings:
        print(g[0], g[1][0], g[1][1], g[1][0] / g[1][1])


def ppe_use_and_travel_2(ds, src, dest, start_ts):
    ds = session.Session()
    s_ptnts = src['patients']
    s_asmts = src['assessments']
    print(s_asmts.keys())
    s_tests = src['tests']

    if 'filtered_patients' not in dest.keys():
        f_ptnts = dest.create_group('filtered_patients')
        f_asmts = dest.create_group('filtered_assessments')
        f_tests = dest.create_group('filtered_tests')

        # calculate patient first positives
        raw_p_ids = ds.get(s_ptnts['id']).data[:]
        raw_p_acts = ds.get(s_ptnts['assessment_count']).data[:]
        raw_a_pids = ds.get(s_asmts['patient_id']).data[:]
        raw_t_pids = ds.get(s_tests['patient_id']).data[:]

        # filter out anyone without assessments
        patient_filter = raw_p_acts > 0

        print("patient_filter:",
              np.count_nonzero(patient_filter), np.count_nonzero(patient_filter == 0))

        # filter patients
        f_p_ids = ds.get(s_ptnts['id']).create_like(f_ptnts, 'id')
        f_p_ids.data.write(ds.apply_filter(patient_filter, raw_p_ids))

        # filter out any orphaned assessments
        with utils.Timer("fk in pk"):
            assessment_filter = persistence.foreign_key_is_in_primary_key(raw_p_ids, raw_a_pids)
        print("assessment_filter:",
              np.count_nonzero(assessment_filter), np.count_nonzero(assessment_filter == False))
        f_a_pids = ds.get(s_asmts['patient_id']).create_like(f_asmts, 'patient_id')
        f_a_pids.data.write(ds.apply_filter(assessment_filter, raw_a_pids))
        for k in ('created_at', 'tested_covid_positive'):
            field = ds.get(s_asmts[k]).create_like(f_asmts, k)
            field.data.write(ds.apply_filter(assessment_filter, ds.get(s_asmts[k]).data[:]))

        # filter out any orphaned tests
        test_filter = persistence.foreign_key_is_in_primary_key(raw_p_ids, raw_t_pids)
        print("test_filter:",
              np.count_nonzero(test_filter), np.count_nonzero(test_filter == False))
        f_t_pids = ds.get(s_tests['patient_id']).create_like(f_tests, 'patient_id')
        f_t_pids.data.write(ds.apply_filter(test_filter, raw_t_pids))

    else:
        f_ptnts = dest['filtered_patients']
        f_asmts = dest['filtered_assessments']
        f_tests = dest['filtered_tests']
        f_p_ids = ds.get(f_ptnts['id'])
        f_a_pids = ds.get(f_asmts['patient_id'])
        f_t_pids = ds.get(f_tests['patient_id'])

    # calculate the shared set of indices for assessments / tests back to patients
    with utils.Timer("get_shared_index"):
        p_inds, a_pinds, t_pinds = ds.get_shared_index((f_p_ids, f_a_pids, f_t_pids))
    print(max(p_inds.max(), a_pinds.max(), t_pinds.max()))

    # now filter only assessments with positive test results
    pos_asmt_tests = ds.get(f_asmts['tested_covid_positive']).data[:] == 3
    print("old tests positive:",
          np.count_nonzero(pos_asmt_tests), np.count_nonzero(pos_asmt_tests == False))

    # now filter only tests with positive test results

    s_asmts = src['assessments']
    a_cats = ds.get(f_asmts['created_at'])
    asmt_filter = a_cats.data[:] >= start_ts
    print(np.count_nonzero(asmt_filter), len(asmt_filter))
    raw_a_cats = ds.apply_filter(asmt_filter, a_cats.data[:])
    a_days = np.zeros(len(raw_a_cats), dtype=np.int32)
    start_dt = datetime.fromtimestamp(start_ts)
    for i_r in range(len(raw_a_cats)):
        a_days[i_r] = (datetime.fromtimestamp(raw_a_cats[i_r]) - start_dt).days
    print(sorted(utils.build_histogram(a_days)))


if __name__ == '__main__':
    datastore = DataStore()
    src_file = '/home/ben/covid/ds_20200901_full.hdf5'
    dest_file = '/home/ben/covid/ds_20200901_ppe.hdf5'
    with h5py.File(src_file, 'r') as src_data:
        with h5py.File(dest_file, 'w') as dest_data:
            start_timestamp = datetime.timestamp(datetime(2020, 6, 12))
            ppe_use_and_travel_2(datastore, src_data, dest_data, start_timestamp)
