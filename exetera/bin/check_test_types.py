
from collections import defaultdict

import numpy as np

from exetera.core.dataset import Dataset
from exetera.core import utils

filename = "/home/ben/covid/covid_test_export_20200720040016.csv"

with open(filename) as f:
    ds = Dataset(f)
    print(ds.names_)


test_types = defaultdict(int)

hgram = sorted([(v[1], v[0]) for v in utils.build_histogram(ds.field_by_name("mechanism"))],
               reverse=True)
for e in hgram[:100]:
    print(e)

def search_for_substring(hgram, substring, exclusions=tuple()):
    blood_count = 0
    hfilter = np.zeros(len(hgram), dtype=np.bool)
    for ie, e in enumerate(hgram):
        entry = e[1].lower()
        if substring in entry and entry not in exclusions:
            blood_count += e[0]
            hfilter[ie] = True
    return blood_count, hfilter


def search_for_substrings(hgram, substrings, exclusions=tuple()):
    count = 0
    hfilter = np.zeros(len(hgram), dtype=np.bool)
    for ie, e in enumerate(hgram):
        entry = e[1].lower()
        if entry not in exclusions:
            all = True
            for s in substrings:
                if s not in entry:
                    all = False
                    break
            if all:
                count += e[0]
                hfilter[ie] = True
    return count, hfilter


def print_unfiltered(hgram, filter_):
    for ih, h in enumerate(hgram):
        if filter_[ih]:
            print(h)

def sum_unfiltered(hgram, filter_):
    unfiltered = 0
    filtered = 0
    for ih, h in enumerate(hgram):
        if filter_[ih] == True:
            unfiltered += h[0]
        else:
            filtered += h[0]
    return unfiltered, filtered



def filter_standard_entries(hgram, standard_entries):
    filter_ = np.zeros(len(hgram), dtype=np.bool)
    count = 0
    for ih, v in enumerate(hgram):
        if v[1] in standard_entries:
            count += v[0]
            filter_[ih] = True
    return count, filter_


print()




antigen_exclusions = ('nose_throat_swab', 'throat_swab', 'nose_swab', 'spit_tube')
antibody_exclusions = ('blood_sample', 'blood_sample_finger_prick', 'blood_sample_needle_draw')

standard_count, standard_filt =\
    filter_standard_entries(hgram, antigen_exclusions + antibody_exclusions)

antibody_strong = ('blood', 'antib', 'anti b', 'anti-b', 'prick', 'antikro', 'anti kro', 'blod', 'all three', 'all tests', 'all of')
antibody_weak = ('prick', 'stick i f', 'finger')
pcr_strong = ('swab', 'swap', 'swob', 'swan', 'tonsil', 'nose', 'throat', 'näsa', 'svalg', 'oral', 'nasoph',
                  'saliva', 'all three', 'all tests', 'all of', 'plasma', 'drive t', 'drivet')
pcr_weak = ('self test', 'self admin', 'home test', 'home admin', 'self', 'home', 'post', 'i did it', 'drive', 'hemma', 'private')

antibody_dict = dict()
for p in antibody_strong:
    count, filt = search_for_substring(hgram, p, antibody_exclusions)
    antibody_dict[p] = count, filt

pcr_dict = dict()
for p in pcr_strong:
    count, filt = search_for_substring(hgram, p, antigen_exclusions)
    pcr_dict[p] = count, filt

weak_pcr_dict = dict()
for p in pcr_weak:
    count, filt = search_for_substring(hgram, p, antibody_exclusions + antigen_exclusions)
    weak_pcr_dict[p] = count, filt

print("searching for 'home' and 'prick'")
home_prick_count, home_prick_filt =\
    search_for_substrings(hgram, ('home', 'prick'), antigen_exclusions)
print(home_prick_count)


antibody_filter = np.zeros(len(hgram), dtype=np.bool)
for k, v in antibody_dict.items():
    print(k, v[0])
    antibody_filter = antibody_filter | v[1]

pcr_filter = np.zeros(len(hgram), dtype=np.bool)
for k, v in pcr_dict.items():
    print(k, v[0])
    pcr_filter = pcr_filter | v[1]

weak_pcr_filter = np.zeros(len(hgram), dtype=np.bool)
for k, v in weak_pcr_dict.items():
    print(k, v[0])
    weak_pcr_filter = weak_pcr_filter | v[1]

total_filter = standard_filt | antibody_filter | pcr_filter | weak_pcr_filter


enumerated = 0
for ih, h in enumerate(hgram):
    if total_filter[ih] == False:
        enumerated += 1
        print(h)
    if enumerated == 100:
        break


both_filter = antibody_filter & (pcr_filter | weak_pcr_filter)
print("both types:", sum_unfiltered(hgram, both_filter))


print(sum_unfiltered(hgram, total_filter))

print("total filter:", np.count_nonzero(total_filter == True), np.count_nonzero(total_filter == False))

