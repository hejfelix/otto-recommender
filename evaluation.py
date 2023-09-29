import numpy as np
from numpy.random import default_rng
from typing import List


class RandomEmbeddings:
    """
    This is a dummy class that returns random items from the vocabulary as recommendations.
    It is used to test the evaluation functions.
    """
    def __init__(self, vocabulary: List[int], seed=None):
        self.vocabulary = vocabulary
        self.rng = default_rng(seed) if seed else default_rng()

    def similar_by_vector(self, query_item: int, topn: int) -> List[tuple[int, float]]:
        return [ (x, 0.0) for x in self.rng.choice(self.vocabulary, topn, replace=False)]


"""
Adapted from https://github.com/fastforwardlabs/session_based_recommenders/blob/c438dd1334fcefc6bedea69b0cd67f779a5de5d3/recsys/metrics.py
"""


def recall_at_k(test, embeddings, k: int = 10) -> float:
    """
    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    ratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        neighbors = embeddings.similar_by_vector(query_item, topn=k)
        # clean up the list
        recommendations = [item for item, score in neighbors]
        # check if ground truth is in the recommedations
        if ground_truth in recommendations:
            ratk_score += 1
    ratk_score /= len(test)
    return ratk_score


def mrr_at_k(test, embeddings, k: int) -> float:
    """
    Mean Reciprocal Rank.

    test must be a list of (query, ground truth) pairs
    embeddings must be a gensim.word2vec.wv thingy
    """
    mrratk_score = 0
    for query_item, ground_truth in test:
        # get the k most similar items to the query item (computes cosine similarity)
        neighbors = embeddings.similar_by_vector(query_item, topn=k)
        # clean up the list
        recommendations = [item for item, score in neighbors]
        # check if ground truth is in the recommedations
        if ground_truth in recommendations:
            # identify where the item is in the list
            rank_idx = np.argwhere(np.array(recommendations) == ground_truth)[0][0] + 1
            # score higher-ranked ground truth higher than lower-ranked ground truth
            mrratk_score += 1 / rank_idx
    mrratk_score /= len(test)
    return mrratk_score
