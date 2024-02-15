import numpy as np
from model_utils import train_test_Ridge, train_test_LR

# pre-define the affective norms for the 9 emotions (from the NRC Lexicon)
affective_norms = [[0.122, 0.83, 0.604],  # angry 0
                   [1, 0.735, 0.772],  # happy 1
                   [0.908,0.931,0.709],  # excited 2
                   [0.225, 0.333, 0.149],  # sad 3
                   [0.08, 0.651, 0.255],  # frustrated 4
                   [0.051, 0.773, 0.274],  # disgusted 5
                   [0.083, 0.482, 0.278],  # fearful 6
                   [0.784, 0.855, 0.539],  # surprised 7
                   [0.469, 0.184, 0.357]  # neutral 8
                   ]


def entropy(model, data):
    pred_prob = model.predict_proba(data)
    ent = np.array([-np.sum(vec * np.log(vec)) for vec in pred_prob])
    return ent


def least_confidence(model, data):
    y_pred_pro = model.predict_proba(data)
    max_prob = y_pred_pro.max(axis=1)
    return max_prob


def maximum_mdl_change(model, unlabeled_X):
    y_pred_pro = model.predict_proba(unlabeled_X)

    def expected_mdl_change(y_pred, x):
        return np.sum((1-y_pred)*y_pred) * np.linalg.norm(x)
    mdl_change = [expected_mdl_change(y_pred_pro[i], unlabeled_X[i]) for i in range(unlabeled_X.shape[0])]
    return mdl_change


def exp_mdl_change_max(unlabeled_X, pool_pred_Y, labeled_X, labeled_Y):
    exp_mdl_change = np.zeros([unlabeled_X.shape[0],])
    for k in range(4):
        # construct 4 models using bootstrap
        boots_ids = np.random.choice(range(labeled_X.shape[0]),labeled_X.shape[0])
        model = train_test_Ridge(labeled_X[boots_ids], labeled_Y[boots_ids],
                                 None, None, cross_validation=True)
        pred_k = model.predict(unlabeled_X)
        exp_mdl_change += np.linalg.norm(pred_k-pool_pred_Y, axis=1)*np.linalg.norm(unlabeled_X, axis=1)
    return exp_mdl_change


def query_by_committee_regression(pool_X, labeled_X, labeled_Y):
    committee_pred = np.zeros([pool_X.shape[0],4, labeled_Y.shape[1]])
    for k in range(4):
        # construct 4 models using bootstrap
        boots_ids = np.random.choice(range(labeled_X.shape[0]), labeled_X.shape[0])
        model = train_test_Ridge(labeled_X[boots_ids,:], labeled_Y[boots_ids],
                                 None, None, cross_validation=True)
        committee_pred[:,k] = model.predict(pool_X)
    pred_var = np.sum(np.var(committee_pred, axis=1), axis=1) # sum the variance on each task
    return pred_var


def compute_cross_task_inconsistency(cat_y, dim_y, scale_min, scale_max,
                                     selected_class=None, selected_dim=None):
    selected_affective_norms = np.array(affective_norms)
    if selected_class is not None:
        selected_affective_norms = selected_affective_norms[selected_class]
    if selected_dim is not None:
        selected_affective_norms = selected_affective_norms[:, selected_dim]
    mapped_dim_y = np.matmul(cat_y, selected_affective_norms)
    mapped_dim_y = mapped_dim_y * (scale_max - scale_min) + scale_min
    inconsistency = np.linalg.norm(mapped_dim_y - dim_y, axis=-1)
    return mapped_dim_y, inconsistency


def select_samples_DEE(target_pool_source_y, target_pool_x, target_pool_y, target_model,
                       selected_target_id, AL_approach:int, distance_mat_x=None,
                       selected_class=None, selected_dim=None, clip_val:list=None):
    # For the target task of dimensional emotion estimation
    unselected_target_id = np.setdiff1d(np.arange(target_pool_x.shape[0]), selected_target_id)
    if distance_mat_x is None:
        distance_mat_x = np.linalg.norm(
            target_pool_x[selected_target_id, None, ...] - target_pool_x[unselected_target_id], axis=-1)
    if AL_approach == 0:
        next_id = np.random.choice(unselected_target_id, 1)
    elif AL_approach == 1:
        # iGS
        pool_target_pred = target_model.predict(target_pool_x[unselected_target_id])
        distance_mat_y = np.abs(target_pool_y[selected_target_id, None, ...] - pool_target_pred)
        if len(distance_mat_y.shape) > 2:
            # Multi-dimensional emotion estimation
            distance_mat_y = np.prod(distance_mat_y, axis=-1)
        distance_mat = distance_mat_x*distance_mat_y
        min_dist = np.min(distance_mat, axis=0)
        next_id = unselected_target_id[np.argmax(min_dist)]
    elif AL_approach in range(2, 5):
        pool_source_y = target_pool_source_y[unselected_target_id]
        pool_target_pred = target_model.predict(target_pool_x[unselected_target_id])
        mapped_target_y, inconsistency = compute_cross_task_inconsistency(pool_source_y, pool_target_pred,
                                              scale_min=clip_val[0], scale_max=clip_val[1],
                                              selected_class=selected_class, selected_dim=selected_dim)
        if AL_approach == 2:
            next_id = unselected_target_id[np.argmax(inconsistency)]
        else:
            # AL_approach 3: GSx-incons-mul, 4: iGS-incons-mul
            if AL_approach == 4:
                distance_mat_y = np.abs(target_pool_y[selected_target_id, None, ...] - pool_target_pred)
                if len(distance_mat_y.shape)>2:
                    distance_mat_y = np.linalg.norm(distance_mat_y, axis=-1)
                distance_mat_x *= distance_mat_y
            min_dist = np.min(distance_mat_x, axis=0)
            inconsistency_dist = min_dist*inconsistency
            next_id = unselected_target_id[np.argmax(inconsistency_dist)]
    elif AL_approach == 5:
        # cross-task iGS
        source_cls_label = np.argmax(target_pool_source_y, axis=1)
        unselected_source_cls = source_cls_label[unselected_target_id]
        selected_source_cls = source_cls_label[selected_target_id]
        pool_target_pred = target_model.predict(target_pool_x[unselected_target_id])
        next_id = None
        cur_max_dist = 0
        for unique_cls in np.unique(unselected_source_cls):
            # each unselected sample is only compared with labeled samples that belong to the same emotion category
            if np.sum(selected_source_cls==unique_cls) == 0:
                # if there is no labeled sample with predicted emotion category 'unique_cls',
                # calculate the distance between unlabeled samples with all the labeled data
                distance_mat_y = np.abs(
                    target_pool_y[selected_target_id, None, ...] - pool_target_pred[unselected_source_cls==unique_cls])
                distance_mat = distance_mat_x[:, unselected_source_cls == unique_cls]
            else:
                distance_mat_y = np.abs(target_pool_y[selected_target_id[selected_source_cls==unique_cls], None, ...] -
                                        pool_target_pred[unselected_source_cls==unique_cls])
                distance_mat = distance_mat_x[selected_source_cls == unique_cls][:, unselected_source_cls == unique_cls]
            if len(distance_mat_y.shape) > 2:
                # Multi-dimensional emotion estimation
                distance_mat_y = np.prod(distance_mat_y, axis=-1)
            distance_mat *= distance_mat_y
            assert len(distance_mat.shape)==2
            min_dist = np.min(distance_mat, axis=0)
            if next_id is None or np.max(min_dist)>cur_max_dist:
                cur_max_dist = np.max(min_dist)
                next_id = unselected_target_id[unselected_source_cls==unique_cls][np.argmax(min_dist)]
    elif AL_approach == 7:
        # query-by-committee
        committee_var = query_by_committee_regression(target_pool_x[unselected_target_id],
                                    target_pool_x[selected_target_id], target_pool_y[selected_target_id])
        next_id = unselected_target_id[np.argmax(committee_var)]
    elif AL_approach == 8:
        # maximum model change
        pool_target_pred = target_model.predict(target_pool_x[unselected_target_id])
        expected_model_change = exp_mdl_change_max(target_pool_x[unselected_target_id], pool_target_pred,
                                                   target_pool_x[selected_target_id], target_pool_y[selected_target_id])
        next_id = unselected_target_id[np.argmax(expected_model_change)]
    elif AL_approach == 9:
        # rank combination
        pool_target_pred = target_model.predict(target_pool_x[unselected_target_id])
        distance_mat_y = np.abs(target_pool_y[selected_target_id, None, ...] - pool_target_pred)
        rank_dimV = np.argsort(np.argsort(-np.min(distance_mat_y[..., 0] * distance_mat_x, axis=0)))
        rank_dimA = np.argsort(np.argsort(-np.min(distance_mat_y[..., 1] * distance_mat_x, axis=0)))
        rank_dimD = np.argsort(np.argsort(-np.min(distance_mat_y[..., 2] * distance_mat_x, axis=0)))
        combined_rank = rank_dimV + rank_dimA + rank_dimD
        next_id = unselected_target_id[np.argmin(combined_rank)]
    else:
        raise ValueError('no AL approach found')
    selected_target_id = np.append(selected_target_id, next_id)
    return selected_target_id


def select_samples_CEC(target_pool_source_y, target_pool_x, target_model, selected_target_id, AL_approach:int,
                       distance_mat_x=None, selected_class=None, selected_dim=None, clip_val:list=None):
    # target_pool_source_y: probability for each emotion category
    # For the target task of categorical emotion classification
    unselected_target_id = np.setdiff1d(np.arange(target_pool_x.shape[0]), selected_target_id)
    if distance_mat_x is None:
        distance_mat_x = np.linalg.norm(
            target_pool_x[selected_target_id, None, :] - target_pool_x[unselected_target_id], axis=-1)
    if AL_approach == 0:
        next_id = np.random.choice(unselected_target_id, 1)
    elif AL_approach == 1:
        # entropy
        target_pool_ent = entropy(target_model, target_pool_x[unselected_target_id])
        next_id = unselected_target_id[np.argmax(target_pool_ent)]
    elif AL_approach == 2:
        # least confidence
        target_pool_confidence = least_confidence(target_model, target_pool_x[unselected_target_id])
        next_id = unselected_target_id[np.argmin(target_pool_confidence)]
    elif AL_approach in range(4, 7):
        # cross-task inconsistency
        pred_prob = target_model.predict_proba(target_pool_x[unselected_target_id])
        mapped_target_pred, inconsistency = compute_cross_task_inconsistency(pred_prob,
                                                 target_pool_source_y[unselected_target_id],
                                                 scale_min=clip_val[0], scale_max=clip_val[1],
                                                 selected_class=selected_class, selected_dim=selected_dim)
        if AL_approach == 4:
            # CTIAL
            next_id = unselected_target_id[np.argmax(inconsistency)]
        elif AL_approach == 5:
            # LC-CTIAL
            target_pool_confidence = least_confidence(target_model, target_pool_x[unselected_target_id])
            next_id = unselected_target_id[np.argmax(inconsistency/target_pool_confidence)]
        else:
            # Ent-CTIAL
            target_pool_ent = entropy(target_model, target_pool_x[unselected_target_id])
            next_id = unselected_target_id[np.argmax(inconsistency * target_pool_ent)]
    elif AL_approach == 8:
        # greedy sampling on the features
        min_dist = np.min(distance_mat_x, axis=0)
        next_id = unselected_target_id[np.argmax(min_dist)]
    elif AL_approach == 9:
        # expected model change maximization
        model_change = maximum_mdl_change(target_model, target_pool_x[unselected_target_id])
        next_id = unselected_target_id[np.argmax(model_change)]
    else:
        raise ValueError('No AL approach found')
    selected_target_id = np.append(selected_target_id, next_id)
    return selected_target_id
