import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def fit_and_cross_validate_score_model(estimator, X, y, threshold=0.5):
    """
    Cross validates and calculates the following average validation scores on the model: Accuracy, ROC AUC, Precision,
    Recall, and F1.

    :param tuple estimator:
    :param pandas.DataFrame X:
    :param y:
    :return:
    """
    # scale data if algorithm requires it
    if estimator[1].__class__.__name__ in ['KNeighborsClassifier', 'LogisticRegression', 'SVM']:
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

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # fit model
        estimator[1].fit(X_train, y_train)
        y_pred = estimator[1].predict_proba(X_val)[:, 1] > threshold

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
    if estimator[1].__class__.__name__ in ['KNeighborsClassifier', 'LogisticRegression', 'SVM']:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X))

    auc_train = []
    auc_val = []
    accuracies_train = []
    accuracies_val = []

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
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


def fit_and_cross_validate_score_roc_auc_xgboost(xgb_model, X, y):
    """

    :param estimator:
    :param X:
    :param y:
    :return: Dictionary containing the name of the model, the training ROC AUC score, and the validation ROC AUC score.
    """
    auc_train = []
    auc_val = []
    accuracies_train = []
    accuracies_val = []

    # split data into 5 folds for cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # fit model
        eval_set = [(X_train, y_train), (X_val, y_val)]
        xgb_model[1].fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='error',  # new evaluation metric: classification error (could also use AUC, e.g.)
            early_stopping_rounds=50,
            verbose=False
        )
        y_pred_train = xgb_model[1].predict(X_train, ntree_limit=xgb_model[1].best_ntree_limit)
        y_pred_val = xgb_model[1].predict(X_val)

        # calculate metrics
        y_scores_train = [y[1] for y in xgb_model[1].predict_proba(X_train, ntree_limit=xgb_model[1].best_ntree_limit)]
        y_scores_val = [y[1] for y in xgb_model[1].predict_proba(X_val, ntree_limit=xgb_model[1].best_ntree_limit)]

        auc_train.append(roc_auc_score(y_train, y_scores_train))
        auc_val.append(roc_auc_score(y_val, y_scores_val))
        accuracies_train.append(accuracy_score(y_train, y_pred_train))
        accuracies_val.append(accuracy_score(y_val, y_pred_val))

    return {
        'Model': xgb_model[0],
        'ROC AUC (Train)':  np.mean(auc_train),
        'ROC AUC (Val)': np.mean(auc_val),
    }


def plot_distribution_pair(d1, d2, d1_label, d2_label, tick_min, tick_max, tick_interval, n_bins):
    """
    Overlays 2 distributions in the same units.

    :param list d1: Numpy array or list containing the first distribution.
    :param list d2: Numpy array or list containing the second distribution.
    :param str d1_label: Legend label for the first distribution.
    :param str d2_label: Legend label for the second distribution.
    :param int tick_min: Value of the lowest tick on the x axis.
    :param int tick_max: Value of the highest tick on the x axis.
    :param int tick_interval: Interval of ticks on x axis.
    :param int n_bins: Number of bins to break the distribution into for plotting purposes.
    """
    # plot distributions
    sns.distplot(d1, bins=n_bins)
    d = sns.distplot(d2, bins=n_bins)

    # format plot
    d.set_xticks(range(tick_min, tick_max, tick_interval))
    plt.legend([d1_label, d2_label])
