from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from data_utils import *
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold


def train_test_Ridge(train_features, train_labels, test_features, test_labels, clip_val=None,
                     reg_strength=1000, cross_validation=False, annotated_labels=None):

    task_res = []
    if cross_validation:
        alpha_candidate = [1, 5, 10, 50, 1e2, 5e2, 1e3, 5e3]  # 备选的alpha范围
        min_score = np.inf
        optim_alpha = alpha_candidate[0]
        for alpha in alpha_candidate:
            model = Ridge(alpha=alpha)
            score = cross_validation_score(model, train_features, train_labels, k=3, task='r')
            if score < min_score:
                min_score = score
                optim_alpha = alpha
        model = Ridge(alpha=optim_alpha)
    else:
        model = Ridge(alpha=reg_strength)
    model.fit(train_features, train_labels)
    if test_features is None:
        return model
    test_pred = model.predict(test_features)
    if len(test_pred.shape) == 1:
        test_pred = np.expand_dims(test_pred, 1)
        test_labels = np.expand_dims(test_labels, 1)
        if annotated_labels is not None and len(annotated_labels.shape) == 1:
            annotated_labels = np.expand_dims(annotated_labels, 1)
    if clip_val is not None:
        test_pred = clip_prediction(test_pred, clip_val[0], clip_val[1])
    if annotated_labels is not None:
        test_pred = np.concatenate([test_pred, annotated_labels])
        test_labels = np.concatenate([test_labels, annotated_labels])
    for i in range(test_labels.shape[1]):
        task_res.extend(compute_rmse_cc(test_pred[:, i], test_labels[:, i]))
    return task_res, model


def train_test_LR(train_features, train_labels, test_features, test_labels,
                  reg_strength=1000, cross_validation=False, annotated_labels=None):

    if test_labels is not None and len(np.unique(train_labels))!=len(np.unique(test_labels)):
        # directly return if not all the classes in the test set are included in the training set
        return 0
    if cross_validation:
        alpha_candidate = [1, 5, 10, 50, 1e2, 5e2, 1e3, 5e3]  # 备选的alpha范围
        max_score = 0
        optim_alpha = alpha_candidate[0]
        for alpha in alpha_candidate:
            model = LogisticRegression(multi_class='auto', solver='liblinear', C=1 / alpha)

            score = cross_validation_score(model, train_features, train_labels, k=3, task='c')
            if score > max_score:
                max_score = score
                optim_alpha = alpha
        model = LogisticRegression(multi_class='auto', class_weight='balanced', solver='liblinear', C=1/optim_alpha)
    else:
        model = LogisticRegression(multi_class='auto', class_weight='balanced', solver='liblinear', C=reg_strength)
    model.fit(train_features, train_labels)
    if test_features is None:
        return model
    test_pred = model.predict(test_features)
    if annotated_labels is not None:
        test_pred = np.concatenate([test_pred, annotated_labels])
        test_labels = np.concatenate([test_labels, annotated_labels])
    bca = balanced_accuracy_score(test_labels, test_pred)
    return bca, model



def cross_validation_score(model, X, y, k=None, task='c'):
    # k is the number of folds in k-fold cross-validation,
    # if k is None, leave-one-out cross-validation is used
    if k is None:
        k = X.shape[0]
    kf = StratifiedKFold(n_splits=k,shuffle=True) if task == 'c' else KFold(n_splits=k,shuffle=True)
    pred_all = []
    y_all = []
    for train_idx, test_idx in kf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        test_pred = model.predict(X[test_idx])
        pred_all.append(test_pred)
        y_all.append(y[test_idx])
    pred_all = np.concatenate(pred_all)
    y_all = np.concatenate(y_all)
    score = 0
    if task == 'c':
        score = balanced_accuracy_score(y_all, pred_all)
    else:
        if len(y_all.shape)==1:
            y_all = np.expand_dims(y_all,1)
            pred_all = np.expand_dims(pred_all,1)
        for i in range(y_all.shape[1]):
            score += compute_rmse_cc(pred_all[:, i], y_all[:, i])[0]
            if np.isnan(score):
                score = np.inf
    return score



