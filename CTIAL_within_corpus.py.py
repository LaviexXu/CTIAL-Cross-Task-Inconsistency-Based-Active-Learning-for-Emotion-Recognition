import numpy as np
from data_utils import get_IEMOCAP_text_audio_features, split_dataset_by_subject, clip_prediction
from model_utils import train_test_Ridge, train_test_LR
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from supervised_active_learning_lib import select_samples_DEE, select_samples_CEC,\
                                            compute_cross_task_inconsistency
from sklearn.model_selection import train_test_split
from itertools import combinations
from data_utils import *


def cross_task_AL_CEC2DEE(al_approach=0, iterations=100):
    # Within-corpus cross-task active learning experiment from CEC to DEE on IEMOCAP dataset
    AL_approach_dict = DEE_AL_approach_dict
    selected_class = [0, 1, 3, 4, 7, ]  # Select sample with angry, happy, sad, frustrated and neutral emotions
    data_df, text_features, acoustic_features = get_IEMOCAP_text_audio_features(
            dir_root='data/IEMOCAP', selected_class=selected_class)
    session_id = data_df['session_id'].values
    unique_session_id = np.unique(session_id)
    pca = PCA(n_components=0.9)
    # PCA on the concatenation of the source and target dataset
    pca_acoustic_features = pca.fit_transform(acoustic_features)
    labels = np.vstack(
        [data_df.Category.values, data_df.val.values, data_df.aro.values, data_df.dom.values]).transpose()
    results = []
    # Each time select 3 sessions as the target dataset, and the rest 2 sessions as the source dataset
    sess_combinations = list(combinations(unique_session_id, 3))
    selected_class = [0, 1, 3, 4, 8]  # the indices of the affective norms of angry, happy, sad, frustrated and neutral emotions

    for rep in range(len(sess_combinations)*3):
        test_session_comb = sess_combinations[rep%len(sess_combinations)]
        pca_source_x, pca_pool_x,_, pool_x, source_y, pool_y = split_dataset_by_subject(
                        session_id, test_session_comb, pca_acoustic_features, acoustic_features, labels)
        source_y = source_y[:, 0]
        pool_y = pool_y[:, 1:]
        np.random.seed(rep)
        # PCA test
        pool_x = pca.fit_transform(pool_x)
        distance_mat_x = np.linalg.norm(pool_x[:, None, :] - pool_x, axis=-1)
        cur_iter_results = []
        # Train the source emotion classification model
        source_model = train_test_LR(pca_source_x, source_y, None, None, cross_validation=True)
        # Obtain the predicted emotion probabilities of the target data
        selected_sample_id = np.random.choice(np.arange(pool_x.shape[0]), 20, replace=False)
        target_pool_source_pred = source_model.predict_proba(pca_pool_x)
        for iteration in range(iterations):
            unselected_sample_id = np.setdiff1d(np.arange(pool_x.shape[0]), selected_sample_id)
            selected_sample = pool_x[selected_sample_id]
            selected_sample_y = pool_y[selected_sample_id]
            if al_approach == 6:
                # The non-AL baseline: NRC Mapping
                # Directly map the predicted emotion probabilities to the dimensional emotions
                mapped_target_y, _ = compute_cross_task_inconsistency(target_pool_source_pred, np.zeros(pool_y.shape),
                                                                      scale_min=1, scale_max=5,
                                                                      selected_class=selected_class)
                for i in range(pool_y.shape[1]):
                    cur_iter_results.extend(compute_rmse_cc(mapped_target_y[:, i], pool_y[:, i]))
                break
            # Evaluate the performance of the emotion estimation model that trained on labeled samples in the target dataset
            # All the samples in the target dataset, including the labeled ones, are used for testing
            test_results, model = train_test_Ridge(selected_sample, selected_sample_y,
                                                   pool_x[unselected_sample_id], pool_y[unselected_sample_id],
                                                   clip_val=[1, 5], cross_validation=True,
                                                   annotated_labels=pool_y[selected_sample_id])
            cur_iter_results.append(test_results)
            # active learning
            if al_approach == -2:
                # Select the sample using Least Confidence on the predictions of the source CEC task
                selected_sample_id = select_samples_CEC(None, pca_pool_x, source_model,
                                                        selected_sample_id, AL_approach=2,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:,unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1, 5])
            else:
                # Within-task AL and CTIAL approaches
                selected_sample_id = select_samples_DEE(target_pool_source_pred, pool_x, pool_y,
                                                        model, selected_sample_id, al_approach,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:, unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1,5])
        results.append(cur_iter_results)
    np.savez('results/CEC2DEE/{}.npz'.format(AL_approach_dict[al_approach]), results=results)
    print('results of {} saved'.format(AL_approach_dict[al_approach]))


def cross_task_AL_DEE2CEC(al_approach=0, iterations=100):
    # Within-corpus cross-task active learning experiment from CEC to DEE on IEMOCAP dataset
    AL_approach_dict = DEE_AL_approach_dict
    selected_class = [0, 1, 3, 4, 7, ]  # Select sample with angry, happy, sad, frustrated and neutral emotions
    data_df, text_features, acoustic_features = get_IEMOCAP_text_audio_features(
        dir_root='data/IEMOCAP', selected_class=selected_class)
    session_id = data_df['session_id'].values
    unique_session_id = np.unique(session_id)
    pca = PCA(n_components=0.9)
    # PCA on the concatenation of the source and target dataset
    pca_acoustic_features = pca.fit_transform(acoustic_features)
    # labels: emotion class, valence, arousal, dominance
    labels = np.vstack(
        [data_df.Category.values, data_df.val.values, data_df.aro.values, data_df.dom.values]).transpose()
    results = []
    # Each time select 3 sessions as the target dataset, and the rest 2 sessions as the source dataset
    sess_combinations = list(combinations(unique_session_id, 3))
    selected_class = [0, 1, 3, 4, 8]  # the indices of the affective norms of angry, happy, sad, frustrated and neutral emotions
    for rep in range(len(sess_combinations)*3):
        test_session_comb = sess_combinations[rep%len(sess_combinations)]
        pca_source_x, pca_pool_x,_, pool_x, source_y, pool_y = split_dataset_by_subject(
                        session_id, test_session_comb, pca_acoustic_features, acoustic_features, labels)
        source_y = source_y[:, 1:]
        pool_y = pool_y[:, 0]
        np.random.seed(rep)
        # PCA test
        pool_x = pca.fit_transform(pool_x)
        distance_mat_x = np.linalg.norm(pool_x[:, None, :] - pool_x, axis=-1)
        cur_iter_results = []
        # Train the source emotion estimation model
        source_model = train_test_Ridge(pca_source_x, source_y, None, None, cross_validation=True)
        # Obtain the predicted emotion probabilities of the target data
        target_pool_source_pred = clip_prediction(source_model.predict(pca_pool_x), 1, 5)
        selected_sample_id = train_test_split(np.arange(pool_y.shape[0]), stratify=pool_y.astype(int), train_size=20)[0]
        for iteration in range(iterations):
            unselected_sample_id = np.setdiff1d(np.arange(pool_x.shape[0]), selected_sample_id)
            # Evaluate the performance of the emotion classification model that trained on labeled samples in the target dataset
            # All the samples in the target dataset, including the labeled ones, are used for testing
            test_results, model = train_test_LR(pool_x[selected_sample_id], pool_y[selected_sample_id],
                                                pool_x[unselected_sample_id], pool_y[unselected_sample_id],
                                                cross_validation=True, annotated_labels=pool_y[selected_sample_id])
            cur_iter_results.append(test_results)
            # active learning
            if al_approach == -2:
                # Select the sample using MTiGS on the predictions of the source DEE task
                selected_sample_id = select_samples_DEE(None, pca_pool_x, target_pool_source_pred,
                                                        source_model, selected_sample_id, AL_approach=1,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:, unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1, 5])
            else:
                # Within-task AL and CTIAL approaches
                selected_sample_id = select_samples_CEC(target_pool_source_pred, pool_x,
                                                        model, selected_sample_id, al_approach,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:, unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1,5])
        results.append(cur_iter_results)
    np.savez('results/DEE2CEC/{}.npz'.format(AL_approach_dict[al_approach]), results=results)
    print('results of {} saved'.format(AL_approach_dict[al_approach]))


CEC_AL_approach_dict = {
    -2: 'source-MTiGS',
    0: 'Rand',
    1: 'Ent',  # Entropy
    2: 'LC',  # Least Confidence
    4: 'CTIAL',
    5: 'LC-CTIAL',
    6: 'Ent-CTIAL',
    8: 'GSx',
    9: 'MMC'
}
DEE_AL_approach_dict = {
    -2: 'source-lc',
    0: 'Rand',
    1: 'MTiGS',
    2: 'CTIAL',
    4: 'MTiGS-CTIAL',
    5: 'CTiGS',  # Cross-task iGS, a variant of MTiGS
    6: 'NRC_Mapping',
    7: 'QBC',  # Query by committee
    8: 'EMCM',  # Expected Model Change Maximization
    9: 'RankComb'  # Rank Combination

}



