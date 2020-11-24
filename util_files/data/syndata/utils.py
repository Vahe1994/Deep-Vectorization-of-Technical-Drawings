import numpy as np

def choose_with_proba(values_probas):
    'Choose value according to corresponding probability'
    values, probas = zip(*values_probas.items())
    assert np.isclose(np.sum(probas), 1.), 'probabilities do not sum to one'
    value = np.random.choice(values, p=probas)
    return value


def renormalize(values_probas, without=None):
    'Exclude keys found in without from values and renormalize probabilities'
    if without is None:
        without = []

    filtered_values_probas = {
        value: proba for value, proba in values_probas.items()
        if value not in without}
    proba_sum = np.sum((proba for proba in filtered_values_probas.values()))
    return {value: proba / proba_sum
            for value, proba in filtered_values_probas.items()}
