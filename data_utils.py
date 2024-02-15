import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import mean_squared_error
from itertools import chain


def get_IEMOCAP_text_audio_features(dir_root='.', selected_class=None):
    # IEMOCAP labels: 0 Ang, 1:happy, 2: excited, 3 sad, 4:frustrated, 5: disgust,
    # 6: surprise,  7:Neu, 8: fear, 9:others
    # selected_class: if None, all classes are used; otherwise, only the selected classes are used
    df = pd.read_csv(os.path.join(dir_root, 'IEMOCAP_text_label.csv'))
    text = df['text'].map(remove_nonverbal_text)
    df['text'] = text
    cat_labels = df['Category']
    cat_labels[cat_labels==2] = 1  # merge 'happy' and 'excited' into one class as in previous work
    df['Category'] = cat_labels
    df = df[text != '']
    wav2vec_features = np.load(os.path.join(dir_root,'wav2vec_features.npz'))['features'] # load audio features
    wav2vec_features = wav2vec_features[text != '']
    txt_features = np.load(os.path.join(dir_root, 'bert_features.npz'))['features'] # load text features
    if selected_class is not None:
        txt_features = txt_features[df.Category.map(lambda x: x in selected_class)]
        wav2vec_features = wav2vec_features[df.Category.map(lambda x: x in selected_class)]
        df = df[df.Category.map(lambda x: x in selected_class)]
        df.Category = df.Category.map(lambda x: selected_class.index(x))
    return df, txt_features, wav2vec_features


def get_VAM_dataset(dir_path='.'):
    f = np.load(os.path.join(dir_path, 'VAM_wav2vec.npz'))
    features, labels, sub_id = f['features'], f['labels'], f['subject_id']
    return features, labels, sub_id


def get_MELD_audio_features(dir_path='.', selected_class=None):
    # anger 0, disgust 1, fear 2, joy 3, sadness 4, surprise 5, neutral 6
    return get_subject_feature_labels(os.path.join(dir_path, 'MELD_train_wav2vec.npz'), selected_class)


def get_subject_feature_labels(npz_file_path, selected_class=None):
    f = np.load(npz_file_path)
    labels = f['labels']
    subject_id = f['subject_id']
    features = f['features']
    if selected_class is not None:
        subject_id = subject_id[np.in1d(labels, selected_class)]
        features = features[np.in1d(labels, selected_class)]
        labels = labels[np.in1d(labels, selected_class)]
        labels = np.array(list(map(lambda x: selected_class.index(x), labels)))
    return subject_id, features, labels


def remove_nonverbal_text(sentence):
    sentence = re.sub(r'\[.*\]', '', sentence)
    # 去除特殊符号
    sentence = sentence.replace('  ', ' ')
    punctuation = r""""#$%&()*+/:;<=>@\^_`{|}~“”"""
    dicts = {i: '' for i in punctuation}
    punc_table = str.maketrans(dicts)
    sentence = sentence.translate(punc_table).strip()
    return sentence


def clip_prediction(pred, min_val, max_val):
    pred[pred < min_val] = min_val
    pred[pred > max_val] = max_val
    return pred


def compute_rmse_cc(pred_y, true_y):
    rmse = mean_squared_error(true_y, pred_y, squared=False)
    corr = np.corrcoef(true_y, pred_y)[0, 1]
    return [rmse, corr]


def split_dataset_by_subject(subject_id, test_subjects, *all_features):
    is_test = np.in1d(subject_id, test_subjects)
    return list(
            chain.from_iterable(
                (features[~is_test], features[is_test]) for features in all_features)
    )

