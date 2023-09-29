import numpy as np
from typing import List
from numpy.random import default_rng

"""
Modified from https://github.com/fastforwardlabs/session_based_recommenders/blob/c438dd1334fcefc6bedea69b0cd67f779a5de5d3/recsys/data.py#L233
the original has Apache License 2.0

No need to reinvent this
"""


def train_test_split(
    session_sequences: List[List[int]], test_size: int = 10000, seed=None
):
    """
    Next Event Prediction (NEP) does not necessarily follow the traditional train/test split.

    Instead training is perform on the first n-1 items in a session sequence of n items.
    The test set is constructed of (n-1, n) "query" pairs where the n-1 item is used to generate
    recommendation predictions and it is checked whether the nth item is included in those recommendations.

    Example:
        Given a session sequence ['045', '334', '342', '8970', '128']
        Training is done on ['045', '334', '342', '8970']
        Testing (and validation) is done on ['8970', '128']

    Test and Validation sets are constructed to be disjoint.
    """

    rng = default_rng(seed) if seed else default_rng()

    ### Construct training set
    # use (1 st, ..., n-1 th) items from each session sequence to form the train set (drop last item)
    train = [sess[:-1] for sess in session_sequences]

    if test_size > len(train):
        print(
            f"Test set cannot be larger than train set. Train set contains {len(train)} sessions."
        )
        return

    ### Construct test and validation sets
    # sub-sample 10k sessions, and use (n-1 th, n th) pairs of items from session_squences to form the
    # disjoint validaton and test sets
    test_validation: List[List[int]] = [sess[-2:] for sess in session_sequences]

    index = rng.choice(range(len(test_validation)), test_size * 2, replace=False)
    test = np.array(test_validation)[index[:test_size]].tolist()
    validation = np.array(test_validation)[index[test_size:]].tolist()

    return train, test, validation
