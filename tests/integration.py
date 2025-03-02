import os
import sys
from collections import Counter
from statistics import median

import pandas as pd
import pytest

from src.clustexts import Clustexts


TEST_DATA = (
    pd.read_csv(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'artifacts',
        'intents.csv'
    ))
    [['text', 'intent']]
    .dropna()
)


def test_clustexts_integration__no_reduce():

    params = {
        'range': (1, 50),
        'min_size': 2,
        'min_gain': 0.001,
        'vectorizer': {
            'max_features': 35000,
            'max_df': 0.25,
            'min_df': 1,
            'use_idf': True
        },
        'reducer': {},
        'verbose': False
    }
    
    df = TEST_DATA.copy()
    cls = Clustexts(**params)
    df['cluster'] = cls(df['text'])

    intent_probabilities = []
    for cluster_id, rows in df.groupby('cluster'):
        intent_dist = Counter(rows.intent)
        mass = sum(intent_dist.values())
        cluster_intent_probabilities = {
            key: val / mass
            for key, val in intent_dist.most_common(1)
        }
        intent_probabilities.extend(
            cluster_intent_probabilities.values()
        )

    assert median(intent_probabilities) >= 0.95
    assert max(df['cluster']) == 9
    assert min(df['cluster']) == 0
    assert len(df['cluster'].unique()) == 10



def test_clustexts_integration__reduce():

    params = {
        'plot_density': False,
        'plot_k': False,
        'show_examples': False,
        'range': (1, 50),
        'min_size': 50,
        'min_gain': 0.001,
        'vectorizer': {
            'max_features': 35000,
            'max_df': 0.25,
            'min_df': 1,
            'use_idf': True
        },
        'reducer': {
            'n_components': 50,
            'n_iter': 20,
            'random_state': 66789
        },
        'verbose': False
    }
    
    df = TEST_DATA.copy()
    cls = Clustexts(**params)
    df['cluster'] = cls(df['text'])
    
    intent_probabilities = []
    for cluster_id, rows in df.groupby('cluster'):
        intent_dist = Counter(rows.intent)
        mass = sum(intent_dist.values())
        cluster_intent_probabilities = {
            key: val / mass
            for key, val in intent_dist.most_common(1)
        }
        intent_probabilities.extend(
            cluster_intent_probabilities.values()
        )
        
    assert median(intent_probabilities) >= 0.95
    assert max(df['cluster']) == 7
    assert min(df['cluster']) == 0
    assert len(df['cluster'].unique()) == 8



if __name__ == '__main__':
    test_clustexts_integration__no_reduce()
    test_clustexts_integration__reduce()



