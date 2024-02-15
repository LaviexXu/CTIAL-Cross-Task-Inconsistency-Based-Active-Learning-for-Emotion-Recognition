import numpy as np
from data_utils import *
from model_utils import train_test_Ridge, train_test_LR
from sklearn.preprocessing import StandardScaler, scale
from sklearn.decomposition import PCA
from traditional_DA import TCA, BDA
from supervised_active_learning_lib import select_samples_DEE, select_samples_CEC
from sklearn.model_selection import train_test_split
from supervised_active_learning_lib import compute_cross_task_inconsistency
import argparse


def cross_corpus_cross_task_AL_CEC2DEE(al_approach=0, iterations=100):
    # Cross-corpus cross-task active learning experiment from CEC on MELD to DEE on IEMOCAP
    AL_approach_dict = DEE_AL_approach_dict
    # Select sample with angry, happy, sad, and neutral emotions
    IEMOCAP_data_df, IEMOCAP_text_features, IEMOCAP_acoustic_features = get_IEMOCAP_text_audio_features(
            dir_root='data/IEMOCAP', selected_class=[0, 1, 3, 7])
    # IEMOCAP: dim_labels: valence, arousal, dominance
    IEMOCAP_dim_labels = np.vstack([IEMOCAP_data_df.val.values, IEMOCAP_data_df.aro.values, IEMOCAP_data_df.dom.values]).transpose()
    IEMOCAP_cat_labels = IEMOCAP_data_df.Category.values
    MELD_subject_id, MELD_features, MELD_labels = get_MELD_audio_features(
              dir_path='data/MELD', selected_class=[0, 3, 4, 6])
    selected_class = [0, 1, 3, 8]

    results = []
    source_x, source_y = MELD_features, MELD_labels
    pool_x, pool_y = IEMOCAP_acoustic_features, IEMOCAP_dim_labels

    # PCA feature dimensionality reduction
    pca = PCA(n_components=0.9)
    # PCA on the concatenation of the source and target dataset
    pca.fit(np.concatenate([source_x, pool_x]))
    source_x = pca.transform(source_x)
    pool_x = pca.transform(pool_x)
    # Use BDA to align the distribution of two datasets and train the source emotion classification model
    tca = BDA(kernel_type='primal', dim=45, mu=0.8)
    da_source_x, da_pool_x = tca.fit(source_x, source_y, pool_x, IEMOCAP_cat_labels)
    source_model = train_test_LR(da_source_x, source_y, None, None, cross_validation=True)
    # Obtain the predicted emotion probabilities of the target data
    target_pool_source_pred = source_model.predict_proba(da_pool_x)
    # PCA on the target dataset
    pool_x = pca.fit_transform(IEMOCAP_acoustic_features)
    distance_mat_x = np.linalg.norm(pool_x[:, None, :] - pool_x, axis=-1)
    for rep in range(30):
        np.random.seed(rep)
        selected_sample_id = np.random.choice(np.arange(pool_x.shape[0]), 20, replace=False)
        cur_iter_results = []
        for iteration in range(iterations):
            unselected_sample_id = np.setdiff1d(np.arange(pool_x.shape[0]), selected_sample_id)
            if al_approach == 6:
                # # The non-AL baseline: NRC Mapping
                # Directly map the predicted emotion probabilities to the dimensional emotions
                mapped_target_y, _ = compute_cross_task_inconsistency(target_pool_source_pred, np.zeros(pool_y.shape),
                                                                      scale_min=1, scale_max=5,
                                                                      selected_class=selected_class)
                for i in range(pool_y.shape[1]):
                    cur_iter_results.extend(compute_rmse_cc(mapped_target_y[:, i], pool_y[:, i]))
                break
            # Evaluate the performance of the emotion estimation model that trained on labeled samples in the target dataset
            # All the samples in the target dataset, including the labeled ones, are used for testing
            test_results, model = train_test_Ridge(pool_x[selected_sample_id], pool_y[selected_sample_id],
                                    pool_x[unselected_sample_id], pool_y[unselected_sample_id], clip_val=[1, 5],
                                    cross_validation=True, annotated_labels=pool_y[selected_sample_id])
            cur_iter_results.append(test_results)
            # active learning
            if al_approach == -2:
                # Select the sample using Least Confidence on the predictions of the source CEC task
                selected_sample_id = select_samples_CEC(None, da_pool_x, source_model,
                                                        selected_sample_id, AL_approach=2,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:,unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1, 5])
            else:
                # Within-task AL and CTIAL approaches
                selected_sample_id = select_samples_DEE(target_pool_source_pred, pool_x, pool_y,
                                                        model, selected_sample_id, al_approach,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:, unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1, 5])
        results.append(cur_iter_results)
    np.savez('results/cross-corpus-CEC2DEE/{}.npz'.format(AL_approach_dict[al_approach]), results=results)
    print('results of {} saved'.format(AL_approach_dict[al_approach]))


def cross_corpus_cross_task_AL_DEE2CEC(al_approach=0, iterations=100):
    # Cross-corpus cross-task active learning experiment from DEE on VAM to CEC on IEMOCAP
    AL_approach_dict = CEC_AL_approach_dict
    # Select sample with angry, happy, sad, frustrated and neutral emotions
    IEMOCAP_data_df, IEMOCAP_text_features, IEMOCAP_acoustic_features = get_IEMOCAP_text_audio_features(
        dir_root='data/IEMOCAP', selected_class=[0, 1, 3, 4, 7])
    IEMOCAP_cat_labels = IEMOCAP_data_df.Category.values
    # load the source dataset VAM
    VAM_features, VAM_labels, VAM_subject_id = get_VAM_dataset(dir_path='data/VAM')
    VAM_labels = VAM_labels * 2 + 3
    selected_class = [0, 1, 3, 4, 8]
    results = []
    source_x = VAM_features
    source_y = VAM_labels
    pool_x = IEMOCAP_acoustic_features
    pool_y = IEMOCAP_cat_labels
    pca = PCA(n_components=0.9)
    # PCA on the concatenation of the source and target dataset
    pca.fit(np.concatenate([source_x, pool_x]))
    source_x = pca.transform(source_x)
    pool_x = pca.transform(pool_x)
    # Use TCA to align the distribution of two datasets and train the source emotion estimation model
    tca = TCA(kernel_type='primal', dim=30)
    da_source_x, da_pool_x = tca.fit(source_x, pool_x, )
    source_model = train_test_Ridge(da_source_x, source_y, None, None, cross_validation=True)
    # Obtain the predicted emotion probabilities of the target data
    target_pool_source_pred = source_model.predict(da_pool_x)
    # PCA on the target dataset
    pool_x = pca.fit_transform(IEMOCAP_acoustic_features)
    distance_mat_x = np.linalg.norm(pool_x[:, None, :] - pool_x, axis=-1)
    for rep in range(30):
        np.random.seed(rep)

        selected_sample_id = train_test_split(np.arange(pool_y.shape[0]), stratify=pool_y.astype(int), train_size=20)[0]
        cur_iter_results = []
        for iteration in range(iterations):
            unselected_sample_id = np.setdiff1d(np.arange(pool_x.shape[0]), selected_sample_id)
            # Evaluate the performance of the emotion estimation model that trained on labeled samples in the target dataset
            # All the samples in the target dataset, including the labeled ones, are used for testing
            test_results, model = train_test_LR(pool_x[selected_sample_id], pool_y[selected_sample_id],
                                                pool_x[unselected_sample_id], pool_y[unselected_sample_id],
                                                cross_validation=True, annotated_labels=pool_y[selected_sample_id])
            # active learning
            if al_approach == -2:
                # Select the sample using MTiGS on the predictions of the source DEE task
                selected_sample_id = select_samples_DEE(None, da_pool_x, target_pool_source_pred, source_model,
                                                        selected_sample_id, AL_approach=1,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:,unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1, 5])
            else:
                # Within-task AL and CTIAL approaches
                selected_sample_id = select_samples_CEC(target_pool_source_pred, pool_x, model,
                                                        selected_sample_id, al_approach,
                                                        distance_mat_x=distance_mat_x[selected_sample_id][:, unselected_sample_id],
                                                        selected_class=selected_class, clip_val=[1, 5])
            cur_iter_results.append(test_results)
        results.append(cur_iter_results)
    np.savez('cross-corpus-DEE2CEC/{}.npz'.format(AL_approach_dict[al_approach]), results=results)
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
