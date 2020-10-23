from collections import defaultdict

import numpy as np

import h5py

import exetera
from exetera.core.session import Session
from exetera.core import utils

connective = {'a', 'i', 'in', 'a', 'of', 'the', 'to', 'the', 'but', 'on', 'for', 'is', 'not', 'had', 'with', 'no',
              'it', 'as', 'at', 'am', 'was', 'has', 'and', 'be', 'all', 'and', 'so', 'or', 'have', 'like', 'been', 'on',
              'off', 'think', 'all', 'time', 'bit', 'feel', 'lot', 'feel', 'couple', 'day', 'days', 'morning', 'would',
              'my', '-', 'och', 'still', 'from', 'this', 'now', '&', 'that', 'when', 'after', 'up', 'just', 'are',
              'which', 'also', 'more', 'some', 'since', 'then', 'under', 'over', 'by', 'me', 'today', 'yesterday',
              'last', 'an', 'only', "i'm", "i’m", 'again', 'if', 'than', 'week', 'get', 'out', 'getting', 'worse',
              'better', 'symptoms', 'feeling', 'very', 'mild', 'slight', 'left', 'right', 'weeks', 'night', 'feels'
              'slightly', 'bad', 'normal', 'around', 'about', 'started', 'having', '/', 'down', 'men', 'before', 'much'
              'test', 'low', 'being' 'got', 'due', "i’ve", 'lite', 'sometimes', 'too', 'they', 'feels', "it’s", 'one',
              'felt', 'severe', 'much', 'taking', 'being', 'med', 'got', 'high', 'march', 'can', '2', '3', '4', 'do',
              'other', 'could', 'even', 'going', 'less', 'during', 'ago', 'both', 'area', 'well', 'and', 'andi', 'som',
              'really', '5', 'though', 'may', 'first', 'two','every', 'usual', 'light', 'occasional', 'little', 'probably',
              'quite', 'times', 'hours', 'gone', "it's", 'red', 'test', 'upper', 'tingling', 'small', 'there', 'any',
              'almost', 'through', 'related', 'go', 'unusual', 'will', 'problems', 'although', 'because', 'these', '1',
             'were', 'you', 'especially', 'never', 'maybe', 'post', 'most', 'few', 'raised', 'seems', 'strange', 'exercise',
              "can't", 'same', 'into', "i've", 'type', 'issues', 'new', 'something', 'possibly', 'diagnosed', 'plus',
              'ongoing', 'went', 'ok', 'hay', '6', 'causing', 'did', 'plus', 'gp', 'virus', 'work', "can't", 'evening',
              'months', 'without', 'following', 'possible', "don't", 'normally', 'he', 'take', 'generally', 'what', '10',
              'increased', 'bed', 'we', 'rate', "can't", "don't", 'general', 'covid', "don’t", "can’t", "?", "fine",
              'home', 'odd', 'etc', 'where', 'doing', 'lots', 'them', 'often', 'things', 'returned', 'keep', '7', 'need',
              'unwell', 'doctor', 'despite', 'experiencing', 'its','starting', 'v', 'medication', 'she', 'paracetamol',
              'wierd', 'always', 'suspected', 'past', 'sure', 'while', 'currently', 'likely', 'took', 'slightly', 'come',
              'seem', 'hour', 'problem', 'diagnose', 'relate', 'improve', 'cause', 'return', 'recover', 'make', 'year',
               'thing', 'recover', "'", 'level', 'know', 'usually', 'see'}

# swedish = {'på', 'jag'}

keywords = {'runny', 'nose', 'pins', 'needles', 'shortness', 'breath', 'throat', 'sense', 'smell', 'pain', 'appetite',
            'sneezing', 'taste', 'cold', 'glands', 'neck', 'sore', 'back', 'swollen'}


cardio = ['heart', 'pulse', 'cardio', 'tachycardia', 'arrhythmia', 'ecg', 'pounding', 'fluttering', 'beating', 'racing', 'skipping', 'beat', 'beating', 'beats',
          'palpations', 'palpitations', 'ectopic', 'erratic', 'rapid']
swelling = ['swollen', 'swelling', 'swell']
hearing = ['hearing']

def replace_multi_with_str(replace_chars, text):
    replacement = []
    for i in range(len(text)):
        if text[i] in replace_chars:
            replacement.append(" ")
        else:
            replacement.append(text[i])
    return "".join(replacement)

# Find overall counts where words appear in the text
def counts_from_full_entries(starts, ends, text, words_to_check):
    total_count = 0
    for i in range(len(starts)):
        substrs = text[starts[i]:ends[i]].tobytes().decode()
        # if ' - ' in substrs:
        #     print(substrs)
        substrs = replace_multi_with_str("#!,\"(){}[].:;", substrs)
        substrs = [s_.strip() for s_ in substrs.split() if len(s_) > 0]
        for s in substrs:
            if s in words_to_check:
                total_count += 1
                break
    print(total_count)

print(exetera.__version__)


with h5py.File('/home/ben/covid/ds_20200901_full.hdf5', 'r') as hf:
    with h5py.File('/home/ben/covid/ds_20200901_othersymp.hdf5', 'w') as tmp:
        s = Session()
        print([k for k in hf['patients'].keys() if 'result' in k])

        old_test = s.get(hf['patients']['max_assessment_test_result']).data[:]
        new_test = s.get(hf['patients']['max_test_result']).data[:]
        test_results = np.where((old_test == 3) | (new_test == 4), 2, 0)
        test_results = np.where((test_results == 0) & ((old_test == 2) | (new_test == 3)), 1, test_results)
        p_test_results = s.create_numeric(tmp, 'p_test_results', 'int8')
        p_test_results.data.write(test_results)
        print("overall tests:", np.unique(test_results, return_counts=True))

        other = s.get(hf['assessments']['other_symptoms'])
        cc = s.get(hf['assessments']['country_code']).data[:]
        otherstart = other.indices[:-1]
        otherend = other.indices[1:]
        ofilter = otherend - otherstart > 0
        print("ofilter:", ofilter.sum(), len(ofilter))
        cfilter = cc == b"GB"
        print("cfilter:", cfilter.sum(), len(cfilter))
        filter_ = ofilter & cfilter
        print("filter_:", filter_.sum(), len(filter_))

        filt_asmt = tmp.create_group('filt_assessments')
        filt_other_symptoms = other.create_like(filt_asmt, 'other_symptoms')
        s.apply_filter(filter_, other, filt_other_symptoms)
        patient_id = s.get(hf['assessments']['patient_id'])
        filt_patient_id = patient_id.create_like(filt_asmt, 'patient_id')
        s.apply_filter(filter_, patient_id, filt_patient_id)
        print('filtered symptoms len =', len(filt_other_symptoms.data))

        with utils.Timer("merging test_results"):
            p_to_a = s.create_numeric(tmp, 'p_to_a', 'int64')
            a_test_results = s.create_numeric(tmp, 'a_test_results', 'int8')
            s.ordered_merge_left(left_on=s.get(tmp['filt_assessments']['patient_id']),
                                 right_on=s.get(hf['patients']['id']), left_field_sources=(p_test_results,),
                                 left_field_sinks=(a_test_results,), left_to_right_map=p_to_a, right_unique=True)
        print(len(a_test_results.data))
        print(np.unique(a_test_results.data[:], return_counts=True))

        a_test_results_ = a_test_results.data[:]
    #     filtered_test_results = test_results[filter_]
    #     print("filtered tests:", np.unique(filtered_test_results, return_counts=True))

        indices, text = s.apply_filter(filter_, other)
        istart = indices[:-1]
        iend = indices[1:]
        print(len(indices), len(text))

import spacy
nlp = spacy.load("en_core_web_sm")

neg_words = defaultdict(int)
pos_words = defaultdict(int)

for i in range(len(istart)):
    if a_test_results_[i] == 1:
        words = neg_words
    elif a_test_results_[i] == 2:
        words = pos_words
    else:
        continue

    substrs = text[istart[i]:iend[i]].tobytes().decode()
    substrs = replace_multi_with_str("#!,\"(){}[].:;", substrs)
    substrs = [s_.strip() for s_ in substrs.split() if len(s_) > 0]
    for s_ in substrs:
        words[s_.lower()] += 1

print(len(neg_words))
print(len(pos_words))


neg_lwords = defaultdict(int)
pos_lwords = defaultdict(int)
neg_lwordlist = None
pos_lwordlist = None

for lwords in (neg_lwords, pos_lwords):
    for wk, wv in words.items():
        doc = nlp(wk)
        token = [w.lemma_.lower() if w.lemma_ != "-PRON-" else w.lower_ for w in doc]
        lwords[token[0]] += wv
    print(len(lwords))

    lwordlist = sorted(list(lwords.items()), key=lambda x: (-x[1], x[0]))

    if lwords == neg_lwords:
        neg_lwordlist = lwordlist
    else:
        pos_lwordlist = lwordlist

neg_filteredwordlist = list()
pos_filteredwordlist = list()

for lwordlist in (neg_lwordlist, pos_lwordlist):
    if lwordlist == neg_filteredwordlist:
        filteredwordlist = neg_filteredwordlist
    else:
        filteredwordlist = pos_filteredwordlist

    for w in lwordlist:
        if w[0] not in connective:
            filteredwordlist.append(w)

    for w in range(200):
        print("{:}, {}, {}, {:3g}%".format(w+1, filteredwordlist[w][0],
                                           filteredwordlist[w][1],
                                           100 * filteredwordlist[w][1] / len(istart)))
