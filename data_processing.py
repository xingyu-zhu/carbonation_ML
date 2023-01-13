import pandas as pd
from sklearn import preprocessing
import shap


def read_data(file):
    data = pd.read_excel(file, 0)
    data = data.iloc[:, :]

    return data


def train_data_value(file):
    data = read_data(file)
    data_df = pd.DataFrame(data, columns=data.columns)
    feature_value = data_df.iloc[:, :-1]
    target_value = data_df.iloc[:, -1]
    cols = feature_value.columns
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    # feature_value = min_max_scaler.fit_transform(feature_value)
    feature_value = pd.DataFrame(feature_value, columns=cols)
    feature_train_summary = shap.kmeans(feature_value, 10)

    return feature_value, target_value, feature_train_summary

def test_data_value(file):
    data = read_data(file)
    data_df = pd.DataFrame(data, columns=data.columns)
    feature_value = data_df.iloc[:, :-1]
    target_value = data_df.iloc[:, -1]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    # feature_value = min_max_scaler.fit_transform(feature_value)

    return feature_value, target_value

def SVM_data_processing(file):
    data = read_data(file)
    data_df = pd.DataFrame(data, columns=data.columns)
    feature_value = data_df.iloc[:, :-1]
    target_value = data_df.iloc[:, -1]
    target_value = pd.DataFrame(target_value)
    cols = feature_value.columns
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    feature_value = min_max_scaler.fit_transform(feature_value)
    feature_value = pd.DataFrame(feature_value, columns=cols)

    training_feature_value = feature_value.iloc[:306, :-1]
    training_target_value = target_value.iloc[:306, -1]
    test_feature_value = feature_value.iloc[306:, :-1]
    test_target_value = target_value.iloc[306:, -1]

    feature_train_summary = shap.kmeans(training_feature_value, 10)

    for i in range(13):
        print(data_df.iloc[:, i].min())
        print(data_df.iloc[:, i].max())
        print(data_df.iloc[:, i].mean())

    return training_feature_value, training_target_value, test_feature_value, test_target_value, feature_train_summary
