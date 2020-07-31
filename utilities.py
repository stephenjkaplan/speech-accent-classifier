import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def fit_and_cross_validate_score_model(estimator, X, y):
    """

    :param estimator:
    :param X:
    :param y:
    :return:
    """
    # scale data if algorithm requires it
    if estimator[0] in ['K-Nearest Neighbors', 'Logistic Regression', 'SVM']:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))

    auc = []
    accuracies = []
    precision_false = []
    recall_false = []
    f1_false = []
    precision_true = []
    recall_true = []
    f1_true = []

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # fit model
        estimator[1].fit(X_train, y_train)
        y_pred = estimator[1].predict(X_val)

        # calculate metrics
        y_scores = [y[1] for y in estimator[1].predict_proba(X_val)]
        auc.append(roc_auc_score(y_val, y_scores))
        accuracies.append(accuracy_score(y_val, y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred)
        precision_false.append(precision[0])
        recall_false.append(recall[0])
        f1_false.append(f1[0])
        precision_true.append(precision[1])
        recall_true.append(recall[1])
        f1_true.append(f1[1])

    return {
        'Model': estimator[0],
        'ROC AUC':  np.mean(auc),
        'Accuracy': np.mean(accuracies),
        'Precision (False)': np.mean(precision_false),
        'Recall (False)': np.mean(recall_false),
        'F1 Score (False)': np.mean(f1_false),
        'Precision (True)': np.mean(precision_true),
        'Recall (True)': np.mean(recall_true),
        'F1 Score (True)': np.mean(f1_true)
    }


def fit_and_cross_validate_score_roc_auc(estimator, X, y):
    """

    :param estimator:
    :param X:
    :param y:
    :return:
    """
    # scale data if algorithm requires it
    if estimator[0] in ['K-Nearest Neighbors', 'Logistic Regression', 'SVM']:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))

    auc_train = []
    auc_val = []
    accuracies_train = []
    accuracies_val = []

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # fit model
        estimator[1].fit(X_train, y_train)
        y_pred_train = estimator[1].predict(X_train)
        y_pred_val = estimator[1].predict(X_val)

        # calculate metrics
        y_scores_train = [y[1] for y in estimator[1].predict_proba(X_train)]
        y_scores_val = [y[1] for y in estimator[1].predict_proba(X_val)]

        auc_train.append(roc_auc_score(y_train, y_scores_train))
        auc_val.append(roc_auc_score(y_val, y_scores_val))
        accuracies_train.append(accuracy_score(y_train, y_pred_train))
        accuracies_val.append(accuracy_score(y_val, y_pred_val))

    return {
        'Model': estimator[0],
        'ROC AUC (Train)':  np.mean(auc_train),
        'ROC AUC (Val)': np.mean(auc_val),
    }


def plot_distribution_pair(d1, d2, d1_label, d2_label, tick_min, tick_max, tick_interval, n_bins):
    """

    :param d1:
    :param d2:
    :param d1_label:
    :param d2_label:
    :param tick_min:
    :param tick_max:
    :param tick_interval:
    :param n_bins:
    """
    sns.distplot(d1, bins=n_bins)
    d = sns.distplot(d2, bins=n_bins)
    d.set_xticks(range(tick_min, tick_max, tick_interval))
    plt.legend([d1_label, d2_label])
