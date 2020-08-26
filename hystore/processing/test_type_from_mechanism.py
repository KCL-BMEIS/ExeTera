import numpy as np

from hystore.core import validation as val


def test_type_from_mechanism_v1(datastore, mechanism, mechanism_free,
                                pcr_standard_answers, pcr_strong_inferred, pcr_weak_inferred,
                                antibody_standard_answers, antibody_strong_inferred, antibody_weak_inferred):

    def search_for_substring(text_entries, pattern):
        filt = np.zeros(len(text_entries), np.bool)
        for ie, e in enumerate(text_entries):
            if pattern in e.lower():
                filt[ie] = True
        return filt

    antigen_exclusions = ('nose_throat_swab', 'throat_swab', 'nose_swab', 'spit_tube')
    antibody_exclusions = ('blood_sample', 'blood_sample_finger_prick', 'blood_sample_needle_draw')

    antibody_strong = ('blood', 'antib', 'anti b', 'anti-b', 'prick', 'antikro', 'anti kro', 'blod', 'all three', 'all tests', 'all of')
    antibody_weak = ('prick', 'stick i f', 'finger')
    pcr_strong = ('swab', 'swap', 'swob', 'swan', 'tonsil', 'nose', 'throat', 'näsa', 'svalg', 'oral', 'nasoph',
                      'saliva', 'all three', 'all tests', 'all of', 'plasma', 'drive t', 'drivet')
    pcr_weak = ('self test', 'self admin', 'home test', 'home admin', 'self', 'home', 'post', 'i did it', 'drive', 'hemma', 'private')


    r_mechanism = val.raw_array_from_parameter(datastore, 'mechanism', mechanism)
    pcr_standard_answers.write(np.isin(r_mechanism, (1, 2, 3, 4)))
    antibody_standard_answers.write(np.isin(r_mechanism, (5, 6, 7)))

    r_mechanism_free = val.raw_array_from_parameter(datastore, 'mechanism_free', mechanism_free)

    f_pcr_strong = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in pcr_strong:
        filt = search_for_substring(r_mechanism_free, p)
        f_pcr_strong = f_pcr_strong | filt
    pcr_strong_inferred.write(f_pcr_strong)

    f_pcr_weak = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in pcr_weak:
        filt = search_for_substring(r_mechanism_free, p)
        f_pcr_weak = f_pcr_weak | filt
    pcr_weak_inferred.write(f_pcr_weak)

    f_antibody_strong = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in antibody_strong:
        filt = search_for_substring(r_mechanism_free, p)
        f_antibody_strong = f_antibody_strong | filt
    antibody_strong_inferred.write(f_antibody_strong)

    f_antibody_weak = np.zeros(len(r_mechanism), dtype=np.bool)
    for p in antibody_weak:
        filt = search_for_substring(r_mechanism_free, p)
        f_antibody_weak = f_antibody_weak | filt
    antibody_weak_inferred.write(f_antibody_weak)

    # count_in_exclusion = 0
    # for r in r_mechanism_free:
    #     if r.lower in antigen_exclusions:
    #         count_in_exclusion += 1
    # print(count_in_exclusion)
