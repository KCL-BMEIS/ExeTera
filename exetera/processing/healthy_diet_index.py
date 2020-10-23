import numpy as np


def score_component(fields, maps, fn, length):
    accumulated = np.zeros(length, np.int32)
    for mk, mv in maps.items():
        if mk in fields and mv is not None:
            accumulated = accumulated + mv[fields[mk]]
    return fn(accumulated)


def healthy_diet_index(fields: dict):
    nda = lambda x: np.array(x, dtype=np.float32)

    print(fields)
    fruit_maps = {
        'ffq_fruit': nda([0.0, 0.0, 4.0, 11.2, 28.8, 56.8, 120.0, 280.0, 480.0])
    }
    fruit_fn = lambda x: np.where(x < 22.9, 1, np.where(x < 160, 2, 3))

    veg_maps = {
        'ffq_salad': nda([0.0, 0.0, 4.0, 11.2, 28.8, 56.8, 120.0, 280.0, 480.0]),
        'ffq_vegetables': nda([0.0, 0.0, 4.0, 11.2, 28.8, 56.8, 120.0, 280.0, 480.0])
    }
    veg_fn = lambda x: np.where(x < 80, 1, np.where(x < 240, 2, 3))

    oily_fish_maps = {
        'ffq_oily_fish': nda([0.0, 0.0, 4.5, 12.6, 32.4, 63.9, 102.6, 102.6, 102.6])
    }
    oily_fish_fn = lambda x: np.where(x < 0.0001, 1, np.where(x < 28.6, 2, 3))

    fat_maps = {
        'ffq_cheese_yogurt': nda([0.0, 0.0, 0.64, 1.79, 4.61, 9.1, 19.22, 44.85, 76.88]),
        'ffq_chips': nda([0.0, 0.0, 0.35, 0.97, 2.49, 4.9, 10.36, 24.18, 41.44]),
        'ffq_crisps_snacks': nda([0.0, 0.0, 6.39, 17.88, 45.98, 90.68, 191.58, 447.02, 766.32]),
        'ffq_eggs': None,
        'ffq_fast_food': None,
        'ffq_fibre_rich_breakfast': nda([0.0, 0.0, 0.23, 0.65, 1.66, 3.28, 6.93, 16.18, 27.73]),
        'ffq_fizzy_pop': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.04]),
        'ffq_fruit': nda([0.0, 0.0, 0.01, 0.03, 0.07, 0.14, 0.3, 0.7, 1.2]),
        'ffq_fruit_juice': nda([0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.07, 0.17, 0.29]),
        'ffq_ice_cream': nda([0.0, 0.0, 0.59, 1.65, 4.24, 8.35, 17.65, 41.18, 70.59]),
        'ffq_live_probiotic_fermented': None,
        'ffq_oily_fish':
            nda([0.0, 0.0, 0.675167, 1.890467, 4.8612, 9.587367, 15.3938, 15.3938, 15.3938]),
        'ffq_pasta': None,
        'ffq_pulses': nda([0.0, 0.0, 0.02, 0.05, 0.14, 0.27, 0.56, 1.32, 2.26]),  # beans
        'ffq_red_meat':
            nda([0.0, 0.0, 0.599595, 1.678865, 4.317081, 8.514243, 13.67076, 13.67076, 13.67076]),
        'ffq_red_processed_meat':
            nda([0.0, 0.0, 1.179906, 3.303738, 8.495325, 16.75467, 26.90186, 26.90186, 26.90186]),
        'ffq_refined_breakfast': None,
        'ffq_rice': None,
        'ffq_salad': nda([0.0, 0.0, 0.5, 1.39, 3.58, 7.06, 14.91, 34.78, 59.63]),
        'ffq_sweets': nda([0.0, 0.0, 1.1, 3.08, 7.92, 15.61, 32.98, 76.95, 131.92]),
        'ffq_vegetables': nda([0.0, 0.0, 0.13, 0.36, 0.94, 1.85, 3.9, 9.11, 15.62]),
        'ffq_white_bread': None,
        'ffq_white_fish':
            nda([0.0, 0.0, 0.115315, 0.322882, 0.830269, 1.637475, 2.629185, 2.629185, 2.629185]),
        'ffq_white_fish_battered_breaded':
            nda([0.0, 0.0, 0.479844, 1.343563, 3.454875, 6.813781, 10.94044, 10.94044, 10.94044]),
        'ffq_white_meat':
            nda([0.0, 0.0, 0.408262, 1.143133, 2.939484, 5.797316, 9.308367, 9.308367, 9.308367]),
        'ffq_white_processed_meat':
            nda([0.0, 0.0, 0.393676, 1.102294, 2.834471, 5.590206, 8.975824, 8.975824, 8.975824]),
        'ffq_wholemeal_bread': nda([0.0, 0.0, 0.28, 0.78, 2.0, 3.94, 8.33, 19.44, 33.33])
    }
    fat_fn = lambda x: np.where(x > 127.5, 1, np.where(x > 85, 2, 3))

    nmes_maps = {
        'ffq_cheese_yogurt': nda([0.0, 0.00, 0.35, 0.97, 2.5, 4.94, 10.44, 24.35, 41.75]),
        'ffq_chips': nda([0.0, 0.0, 0.03, 0.08, 0.2, 0.39, 0.83, 1.93, 3.31]),
        'ffq_crisps_snacks': nda([0.0, 0.0, 0.22, 0.62, 1.58, 3.12, 6.6, 15.4, 26.40]),
        'ffq_eggs': None,
        'ffq_fast_food': None,
        'ffq_fibre_rich_breakfast': nda([0.0, 0.0, 1.18, 3.31, 8.51, 16.79, 35.47, 82.76, 141.87]),
        'ffq_fizzy_pop': nda([0.0, 0.0, 0.21, 0.58, 1.49, 2.94, 6.21, 14.5, 24.86]),
        'ffq_fruit': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'ffq_fruit_juice': nda([0.0, 0.0, 0.3, 0.83, 2.13, 4.2, 8.88, 20.72, 35.52]),
        'ffq_ice_cream': nda([0.0, 0.0, 0.87, 2.44, 6.27, 12.36, 26.11, 60.92, 104.43]),
        'ffq_live_probiotic_fermented': None,
        'ffq_oily_fish': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'ffq_pasta': None,
        'ffq_pulses': nda([0.0, 0.0, 0.13, 0.37, 0.95, 1.86, 3.94, 9.19, 15.76]),  # beans
        'ffq_red_meat': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'ffq_red_processed_meat':
            nda([0, 0, 0.14675, 0.4109, 1.0566, 2.08385, 3.3459, 3.3459, 3.3459]),
        'ffq_refined_breakfast': None,
        'ffq_rice': None,
        'ffq_salad': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'ffq_sweets': nda([0.0, 0.0, 2.12, 5.93, 15.26, 30.1, 63.58, 148.36, 254.33]),
        'ffq_vegetables': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'ffq_white_bread': None,
        'ffq_white_fish':
            nda([0.0, 0.0, 0.010378, 0.029059, 0.074723, 0.14737, 0.236622, 0.236622, 0.236622]),
        'ffq_white_fish_battered_breaded':
            nda([0.0, 0.0, 0.000938, 0.002625, 0.00675, 0.013313, 0.021375, 0.021375, 0.021375]),
        'ffq_white_meat': nda([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'ffq_white_processed_meat':
            nda([0.0, 0.0, 0.028088, 0.078647, 0.202235, 0.398853, 0.640412, 0.640412, 0.640412]),
        'ffq_wholemeal_bread': nda([0.0, 0.0, 0.2, 0.56, 1.44, 2.84, 6.0, 14.0, 24.0])
    }
    nmes_fn = lambda x: np.where(x > 90, 1, np.where(x > 60, 2, 3))

    lengths = set()
    for v in fields.values():
        lengths.add(len(v))
    if len(lengths) != 1:
        raise ValueError("All fields must be the same length (field lengths are {})".format(lengths))
    length = list(lengths)[0]

    fruit_score = score_component(fields, fruit_maps, fruit_fn, length)
    veg_score = score_component(fields, veg_maps, veg_fn, length)
    oily_fish_score = score_component(fields, oily_fish_maps, oily_fish_fn, length)
    fat_score = score_component(fields, fat_maps, fat_fn, length)
    nmes_score = score_component(fields, nmes_maps, nmes_fn, length)

    return fruit_score + veg_score + oily_fish_score + fat_score + nmes_score

