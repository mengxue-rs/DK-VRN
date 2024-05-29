"""
This script implements several metrics for
drought monitoring / detection
"""

import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_classification_metrics(y_pred, y_true, metric_str="", f_round=4, threshold=None):
    assert (metric_str in ["log_loss", "acc", "f1", "precis", "recall", "pr_auc", "roc_auc",
                           "mae", "mse", "rmse", "r2",
                           "pr_curve", "roc_curve", "cm", "avg_prob"])
    """
    scalar metrics
    """
    if metric_str == "log_loss":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        return np.round(log_loss(y_true, y_pred), f_round)

    if metric_str == "acc":
        t = 0.5
        if threshold is not None:
            t = threshold
        y_pred = np.array((y_pred[:] > t), dtype=np.int8)
        y_true = y_true.astype(np.int8)
        return np.round(balanced_accuracy_score(y_true, y_pred), f_round)

    if metric_str == "f1":
        t = 0.5
        if threshold is not None:
            t = threshold
        y_pred = np.array((y_pred[:] > t), dtype=np.int8)
        y_true = y_true.astype(np.int8)
        results = np.round(f1_score(y_true, y_pred, average=None), f_round)
        results = np.average(results)
        return results

    if metric_str == "precis":
        t = 0.5
        if threshold is not None:
            t = threshold
        y_pred = np.array((y_pred[:] > t), dtype=np.int8)
        y_true = y_true.astype(np.int8)
        results = np.round(precision_score(y_true, y_pred, average=None), f_round)
        results = results[1]
        return results

    if metric_str == "recall":
        t = 0.5
        if threshold is not None:
            t = threshold
        y_pred = np.array((y_pred[:] > t), dtype=np.int8)
        y_true = y_true.astype(np.int8)
        results = np.round(recall_score(y_true, y_pred, average=None), f_round)
        results = results[1]
        return results

    if metric_str == "pr_auc":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        results = np.round(average_precision_score(y_true, y_pred, average=None, pos_label=1), f_round)
        return results

    if metric_str == "roc_auc":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        results = np.round(roc_auc_score(y_true, y_pred), f_round)
        return results

    if metric_str == "mae":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        results = np.round(mean_absolute_error(y_true, y_pred), f_round)
        return results

    if metric_str == "mse":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        results = np.round(mean_squared_error(y_true, y_pred, squared=True), f_round)
        return results

    if metric_str == "r2":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        results = np.round(r2_score(y_true, y_pred), f_round)
        return results

    if metric_str == "rmse":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        results = np.round(mean_squared_error(y_true, y_pred, squared=False), f_round)
        return results

    """
    matrix metrics
    """
    if metric_str == "pr_curve":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
        return np.round(precision, f_round), np.round(recall, f_round), np.round(pr_auc, f_round)

    if metric_str == "roc_curve":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        return np.round(fpr, f_round), np.round(tpr, f_round), np.round(roc_auc, f_round)

    if metric_str == "class_likelihood_ratios":
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        return np.round(fpr, f_round), np.round(tpr, f_round), np.round(roc_auc, f_round)

    if metric_str == "cm":
        t = 0.5
        if threshold is not None:
            t = threshold
        y_pred = np.array((y_pred[:] > t), dtype=np.int8)
        y_true = y_true.astype(np.int8)
        cm = confusion_matrix(y_true, y_pred)
        return np.round(cm, f_round)

    if metric_str == "avg_prob":
        # only make senses for matrix data and for analysis under local regions
        # input: array of [n_time, n_point]
        # return: array of [n_time] and [n_time]
        # n_time could be 1
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.astype(np.float32)
        assert (len(y_pred.shape) >= 2)
        assert (len(y_true.shape) >= 2)
        assert (y_pred.shape == y_pred.shape)

        y_true = np.mean(y_true, axis=1)
        y_pred = np.mean(y_pred, axis=1)

        return np.round(y_pred, f_round), np.round(y_true, f_round)

