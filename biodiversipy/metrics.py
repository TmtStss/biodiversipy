import pandas as pd
import numpy as np

def compute_average(y_true, y_pred, t):
    """Returns the average number of species observed correctly predicted given a threshold value t"""
    assert t <= 1
    assert t >= 0
    N, C = y_pred.shape
    temp = y_pred[y_true == 1].applymap(lambda x: 1 if x >= t else 0)
    average = temp.values.sum()/N
    return average

def find_t_min(y_true, y_pred, K, rate, t):
    """Returns the minimum threshold t and corresponding average satisfying the condition average <= K. The minimum t is found iteratively, with tuning parameter rate [0-1]"""
    assert rate <= 1
    assert rate >= 0
    assert K > 0
    average = compute_average(y_true, y_pred, t)
    while average <= K:
        t = rate*t
        average = compute_average(y_true, y_pred, t)
    t_min = t/rate
    average = compute_average(y_true, y_pred, t_min)
    return t_min, average

def compute_accuracy(y_true, y_pred, t_min):
    N, C = y_pred.shape
    temp = y_pred[y_true == 1].applymap(lambda x: 1 if x >= t_min else 0)
    return temp.values.sum()/(N*C)

def custom_metric(y_true, y_pred, K, rate=0.99, t=1):
    t_min, average = find_t_min(y_true, y_pred, K, rate, t)
    accuracy = compute_accuracy(y_true, y_pred, t_min)
    return t_min, average, accuracy
