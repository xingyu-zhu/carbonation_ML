"""
This code is jointly developed by the School of Energy and Environment of University of Science and Technology Beijing
  and the Department of Electronic Information Engineering of Hong Kong Polytechnic University.
The paper has been published in the journal Environmental Science and Technology.
Please note that the conclusions presented in the paper are based on the results of the Apple M1 chip training model.
During the writing of the paper, it was found that there would be differences in various indicators (R2, MAE, RMSE) of
  the training model under Apple chips, Intel chips and Nvidia GPU.
We have not tested chips or GPU of the same type but different versions.Please pay attention to this issue
  during development or use.
This minor issue will not have a significant impact on the training results of the model, but may affect the selection
  of the optimized model.
"""

import shap
from data_processing import train_data_value, test_data_value, scaler_data_processing
from model import MLP_predict, RF_predict, SVR_predict
from figure_output import heat_map, SHAP_plot, predict_compare_plot, other_SHAP_plot, test_predict_plot
from auto_param import run_auto_param
from data_statistics import bar_plot
import os

import warnings

warnings.filterwarnings("ignore")

SHAP_plot_save_path = './SHAP_plot/'
feature_SHAP_plot_path = 'All feature SHAP plots/'
heat_map_path = "heatmap/"

"""
Please note that we used an independent paper dataset as the test dataset,
so we did not partition the dataset proportionally.
"""
# The dataset include train dataset
train_data_file = './train_data.xlsx'
# The dataset include test dataset
test_data_file = './test_data.xlsx'
# The dataset include all data
dataset_file = './dataset.xlsx'

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
    train_feature, train_target, feature_train_summary = train_data_value(train_data_file)
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
        if not os.path.exists(SHAP_plot_save_path):
            os.mkdir(SHAP_plot_save_path)
        if not os.path.exists(SHAP_plot_save_path + feature_SHAP_plot_path):
            os.mkdir(SHAP_plot_save_path + feature_SHAP_plot_path)
        if not os.path.exists(SHAP_plot_save_path + heat_map_path):
            os.mkdir(SHAP_plot_save_path + heat_map_path)

        if os.path.exists(SHAP_plot_save_path) and os.path.exists(SHAP_plot_save_path + feature_SHAP_plot_path)\
                and os.path.exists(SHAP_plot_save_path + heat_map_path):
            print("The SHAP plot directory already exists.")
        else:
            print("Successfully created folder, path is: " + SHAP_plot_save_path)
    except Exception as mkdir_error:
        print("Make Directory Error: " + str(mkdir_error))

def Initialization():
    print("Initialize...")
    make_directory()

if __name__ == "__main__":
    Initialization()
    run_model()
