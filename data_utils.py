import numpy as np
from skmultilearn.dataset import load_from_arff
from typing import Tuple, Iterable
import random
import pandas as pd
"""
    This is a generic data_utils file. It has methods to prepare data and to calculate metrics.
"""

def prepare_data_csv(dataset_name, label_position: int):
    data = np.genfromtxt('../datasets/'+dataset_name, delimiter=',', names=True)
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    y = df.iloc[:,label_position]
    X = df.drop(df.columns[[label_position]], axis=1)
    return X, y

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    This method computes the accuracy as the ratio of correctly predicted labels
    to the total number of predictions using the formula:
    accuracy = number of correct predictions / total number of predictions.

    Args:
        y (np.ndarray): Array of true labels.
        y_hat (np.ndarray): Array of predicted labels.

    Returns:
        float: Accuracy of the predictions, a value between 0 and 1 inclusive.

    Raises:
        ValueError: If `y` and `y_hat` do not have the same shape/size.
    """

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the precision of predictions.

    Args:
        y (np.ndarray): Array of true labels.
        y_hat (np.ndarray): Array of predicted labels.

    Returns:
        float: Precision of the predictions, a value between 0 and 1 inclusive.
    """

    hatTrueIndicies = np.nonzero(y_hat)
    realValues = y[hatTrueIndicies]
    try:
        return np.shape(np.nonzero(realValues))[1]/np.shape(hatTrueIndicies)[1]
    except Exception as e:
        return 0

def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the recall of predictions.

    Args:
        y (np.ndarray): Array of true labels.
        y_hat (np.ndarray): Array of predicted labels.

    Returns:
        float: Recall of the predictions, a value between 0 and 1 inclusive.
    """
    # tp / all pos
    truePositiveInd = np.nonzero(y[np.nonzero(y_hat)])
    allPosInd = np.nonzero(y)
    return np.shape(truePositiveInd)[1]/np.shape(allPosInd)[1]

def false_positive_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the fp rate of predictions.

    Args:
        y (np.ndarray): Array of true labels.
        y_hat (np.ndarray): Array of predicted labels.

    Returns:
        float: FPR of the predictions, a value between 0 and 1 inclusive.
    """
    # fp / all neg
    false_positive_ind = np.nonzero((y == 0) & (y_hat == 1))
    all_neg_ind = np.nonzero(y==0)
    return np.shape(false_positive_ind)[1]/np.shape(all_neg_ind)[1]


def roc_curve_pairs(y: np.ndarray, confidence: np.ndarray) -> Iterable[Tuple[float, float]]:
    """
    Find the ROC curve pairs for a number of confidence values

    Args:
        y (np.ndarray): Array of true labels.
        y_hat (np.ndarray): Array of predicted labels.

    Returns:
        A list of tuples where each tuple contains the FPR and the TPR
    """
    confidence_values = np.unique(confidence)
    # confidence_values = np.arange(0, 1.01, 0.01)
    curve_pairs = []
    for threshold in confidence_values:
        y_hat = confidence >= threshold
        TPR = recall(y,y_hat)
        FPR = false_positive_rate(y,y_hat)
        curve_pairs.append((FPR,TPR))
    return curve_pairs


def auc(y: np.ndarray, confidence: np.array) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) for a given set of true labels and confidence scores.

    Args:
        y (np.ndarray): Array of true labels.
        confidence (np.ndarray): Array of confidence scores.

    Returns:
        float: The computed AUC value.
    """
    roc_pairs = roc_curve_pairs(y, confidence)
    FPR_list = np.sort([t[0] for t in roc_pairs])
    TPR_list = np.sort([t[1] for t in roc_pairs])
    return np.trapz(TPR_list,FPR_list)

