import os
import sys
import pytest

import pandas as pd

from src.clustexts import Clustexts


TEXTS = (
    pd.read_csv(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'artifacts',
        'intents.csv'
    ))
    .rename(columns={'TWEET_TEXT': 'text'})
    ['text']
    .dropna()
)


def test_clustexts_integration():
    params = {
        'plot_density': True,
        'plot_k': True,
        'show_examples': True,
        'range': (1, 50),
        'min_size': 2,
        'min_gain': 0.001,
        'vectorizer': {
            'max_features': 35000,
            'max_df': 0.25,
            'min_df': 1,
            'use_idf': True
        },
        'reducer': {
#             'n_components': 200,
#             'n_iter': 20
        }
    }
    cls = Clustexts(**params)
    clusters = cls(TEXTS)
    assert True



"""
PARAMS = {
    'plot_density': True,
    'plot_k': True,
    'show_examples': True,
    'range': (8, 50),
    'min_size': 0,
    'min_gain': 0.03,
    'vectorizer': {
        'max_features': 35000,
        'max_df': 0.5,
        'min_df': 1,
        'use_idf': True
    },
    'reducer': {
        'n_components': 200,
        'n_iter': 20
    }
}
"""

if __name__ == '__main__':
    test_clustexts_integration()



