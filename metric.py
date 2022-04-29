import math
from typing import List, Callable, Union

from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, mean_absolute_error, r2_score, \
    precision_recall_curve, auc, recall_score, confusion_matrix, precision_score, matthews_corrcoef


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:

    hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def recall(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:

    hard_preds = [1 if p > threshold else 0 for p in preds]
    return recall_score(targets, hard_preds)

def precision(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:

    hard_preds = [1 if p > threshold else 0 for p in preds]
    return precision_score(targets, hard_preds)

def mcc(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:

    hard_preds = [1 if p > threshold else 0 for p in preds]
    return matthews_corrcoef(targets, hard_preds)

def sensitivity(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:

    return recall(targets, preds, threshold)


def specificity(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:

    hard_preds = [1 if p > threshold else 0 for p in preds]
    tn, fp, _, _ = confusion_matrix(targets, hard_preds).ravel()
    return tn / float(tn + fp)



def rmse(targets: List[float], preds: List[float]) -> float:

    return math.sqrt(mean_squared_error(targets, preds))


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:

    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'recall':
        return recall

    if metric == 'sensitivity':
        return sensitivity

    if metric == 'specificity':
        return specificity

    raise ValueError(f'Metric "{metric}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:

    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)
