import shap
from data_processing import train_data_value, test_data_value, scaler_data_processing
from model import MLP_predict, RF_predict, SVR_predict
from figure_output import heat_map, SHAP_plot, predict_compare_plot, other_SHAP_plot, test_predict_plot
from auto_param import run_auto_param
from data_statistics import bar_plot
import os

import warnings

warnings.filterwarnings("ignore")

plot_save_path = './SHAP_plot/'
data_file = './data.xlsx'
test_data_file = './test_data.xlsx'
new_data_file = './new_data.xlsx'

def result_plot(model_predict, feature_train_summary, train_feature, train_target, train_data_predict, test_target, test_data_predict):
    heat_map(train_feature)
    test_predict_plot(test_data_predict, test_target)
    shap.initjs()
    SHAP_plot(model_predict, feature_train_summary, train_feature)
    other_SHAP_plot(model_predict, feature_train_summary, train_feature)
    predict_compare_plot(train_target, train_data_predict, test_target, test_data_predict)

#
def run_model():
    print("Data is processing...")
    # train_feature, train_target, test_feature, test_target, feature_train_summary = scaler_data_processing(new_data_file)
    train_feature, train_target, feature_train_summary = train_data_value(data_file)
    test_feature, test_target = test_data_value(test_data_file)
    bar_plot(train_feature)
    # run_auto_param(train_feature, train_target, test_feature, test_target, param, feature_train_summary)
    print("Model is training...")
    model_predict, train_data_predict, test_data_predict, trained_model = MLP_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict = RF_predict(train_feature, train_target, test_feature, test_target)
    # model_predict, train_data_predict, test_data_predict = SVR_predict(train_feature, train_target, test_feature, test_target)
    print("Generate pictures...")
    result_plot(model_predict, feature_train_summary, train_feature, train_target, train_data_predict, test_target, test_data_predict)

# Make a directory to save result picture
def make_directory():
    print("Check Directory...")
    try:
        if os.path.exists(plot_save_path):
            print("The plot directory already exists.")
        else:
            os.mkdir(plot_save_path)
    except Exception as mkdir_error:
        print("Make Directory Error: " + str(mkdir_error))

if __name__ == "__main__":
    make_directory()
    run_model()
