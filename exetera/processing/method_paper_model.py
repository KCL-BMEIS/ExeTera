import numpy as np

from exetera.core import utils


def method_paper_model(ds, symptoms_reader_dict, prediction):

    intercept = -1.19015973
    weights = {'persistent_cough': 0.23186655,
               'fatigue': 0.56532346,
               'delirium': -0.12935112,
               'shortness_of_breath': 0.58273967,
               'fever': 0.16580974,
               'diarrhoea': 0.10236126,
               'abdominal_pain': -0.11204163,
               'chest_pain': -0.12318634,
               'hoarse_voice': -0.17818597,
               'skipped_meals': 0.25902482,
               'loss_of_smell': 1.82895239}

    with utils.Timer("predicting covid by assessment", new_line=True):
        cumulative = np.zeros(len(symptoms_reader_dict['persistent_cough']), dtype='float32')
        for s in symptoms_reader_dict:
            cumulative += symptoms_reader_dict[s][:] * weights[s]
        cumulative += intercept
        prediction.write(cumulative)
