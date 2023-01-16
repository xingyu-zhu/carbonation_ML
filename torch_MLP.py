import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shap
from data_processing import train_data_value, test_data_value, SVM_data_processing
from model import MLP_predict, RF_predict, SVR_predict, find_best_parameters
from figure_output import heat_map, SHAP_summary_plot, predict_plot, other_SHAP_plot, test_predict_plot
from auto_param import run_auto_param
from data_statistics import bar_plot
import warnings

warnings.filterwarnings("ignore")

data_file = './data.xlsx'
test_data_file = './test_data.xlsx'
new_data_file = './new_data.xlsx'

predict_data = [[90, 5, 1, 100, 74, 40, 10, 10, 2, 35, 3, 20], [90, 5, 1, 60, 74, 40, 10, 10, 2, 35, 3, 20],
                [90, 5, 1, 30, 74, 40, 10, 10, 2, 35, 3, 20], [90, 5, 1, 100, 74, 40, 10, 10, 2, 35, 3, 10],
                [90, 5, 1, 60, 74, 40, 10, 10, 2, 35, 3, 10], [90, 5, 1, 30, 74, 40, 10, 10, 2, 35, 3, 10],
                [90, 5, 1, 100, 74, 40, 15, 10, 2, 30, 3, 20], [90, 5, 1, 60, 74, 40, 15, 10, 2, 30, 3, 20],
                [90, 5, 1, 30, 74, 40, 15, 10, 2, 30, 3, 20], [90, 5, 1, 100, 74, 40, 15, 10, 2, 30, 3, 10],
                [90, 5, 1, 60, 74, 40, 15, 10, 2, 30, 3, 10], [90, 5, 1, 30, 74, 40, 15, 10, 2, 30, 3, 10],
                [90, 1, 1, 100, 74, 40, 15, 10, 2, 30, 3, 20], [90, 1, 1, 60, 74, 40, 15, 10, 2, 30, 3, 20],
                [90, 1, 1, 30, 74, 40, 15, 10, 2, 30, 3, 20], [90, 1, 1, 100, 74, 40, 15, 10, 2, 30, 3, 10],
                [90, 1, 1, 60, 74, 40, 15, 10, 2, 30, 3, 10], [90, 1, 1, 30, 74, 40, 15, 10, 2, 30, 3, 10],

                [90, 5, 1, 100, 74, 40, 15, 25, 10, 7, 3, 20], [90, 5, 1, 60, 74, 40, 15, 25, 10, 7, 3, 20],
                [90, 5, 1, 30, 74, 40, 15, 25, 10, 7, 3, 20], [90, 5, 1, 100, 74, 40, 15, 25, 10, 7, 3, 10],
                [90, 5, 1, 60, 74, 40, 15, 25, 10, 7, 3, 10], [90, 5, 1, 30, 74, 40, 15, 25, 10, 7, 3, 10],

                [90, 5, 1, 100, 74, 45, 15, 15, 10, 12, 3, 20], [90, 5, 1, 60, 74, 45, 15, 20, 10, 7, 3, 20],
                [90, 5, 1, 30, 74, 45, 15, 20, 10, 7, 3, 20], [90, 5, 1, 100, 74, 45, 15, 20, 10, 7, 3, 10],
                [90, 5, 1, 60, 74, 45, 15, 20, 10, 7, 3, 10], [90, 5, 1, 30, 74, 45, 15, 20, 10, 7, 3, 10]
                ]


def result_plot(model_predict, feature_train_summary, train_feature, train_target, train_data_predict, test_target, test_data_predict):
    heat_map(train_feature)
    test_predict_plot(test_data_predict, test_target)
    shap.initjs()
    SHAP_summary_plot(model_predict, feature_train_summary, train_feature)
    other_SHAP_plot(model_predict, feature_train_summary, train_feature)
    predict_plot(train_target, train_data_predict, test_target, test_data_predict)

def run_model():
    # train_feature, train_target, test_feature, test_target, feature_train_summary = SVM_data_processing(new_data_file)
    train_feature, train_target, feature_train_summary = train_data_value(data_file)
    test_feature, test_target = test_data_value(test_data_file)
    bar_plot(train_feature)
    # run_auto_param(train_feature, train_target, test_feature, test_target, param, feature_train_summary)
    # find_best_parameters(train_feature, train_target)
    model_predict, train_data_predict, test_data_predict, trained_model = MLP_predict(train_feature, train_target, test_feature, test_target)
    print(model_predict(predict_data))
    # model_predict, train_data_predict, test_data_predict = RF_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict = SVR_predict(train_feature, train_target, test_feature, test_target)
    result_plot(model_predict, feature_train_summary, train_feature, train_target, train_data_predict, test_target, test_data_predict)

if __name__ == "__main__":
    run_model()
