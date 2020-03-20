import numpy as np


def reciprocal_rank(y_true, y_pred):
    rank = 1
    y_true_ = set(y_true)
    for y_p in y_pred:
        if y_p in y_true_:
            break
        rank += 1

    if rank <= len(y_true):
        return 1 / rank
    else:
        return 0


def mean_reciprocal_rank(y_true, y_pred):
    # check shapes

    mrr = 0
    for y_t, y_p in zip(y_true, y_pred):
        mrr += reciprocal_rank(y_t, y_p)

    return np.mean(mrr)


def discounted_cumulative_gain(relevances):
    dcg_value = 0
    for index, relevance in enumerate(relevances):
        dcg_value += ((2 ** relevance - 1) / np.log2(index + 2))

    return dcg_value


def ndcg(y_true, y_pred):
    return discounted_cumulative_gain(y_pred) / discounted_cumulative_gain(y_true)


def ndcg_at_k(y_true, y_pred, k=10):
    return ndcg(y_true[:k], y_pred[:k])