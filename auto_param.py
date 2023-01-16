import time
import shap
import matplotlib.pyplot as plt
from shap import summary_plot

from model import MLP_auto

_solver = 'lbfgs'
_activation = 'relu'
_alpha = 0.001
_random_state = 3
_max_iter = 10000
_warm_start = True

def run_auto_param(train_feature, train_target, test_feature, test_target, param, feature_train_summary):
    param_select_path = './select_param/'

    for layer_number in range(1, 4):
        if layer_number == 1:
            for layer1 in range(1, 101):
                param = (layer1)
                test_score, RMSE_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                if test_score > 0 and RMSE_score < 6:
                    print(param)
            print("layer 1 finish")
        elif layer_number == 2:
            for layer1 in range(1, 101):
                for layer2 in range(1, 101):
                    if layer2 <= layer1:
                        param = (layer1, layer2)
                        test_score, RMSE_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                        if test_score > 0 and RMSE_score < 6:
                            print(param)
            print("layer 2 finish")
        elif layer_number == 3:
            for layer1 in range(1, 101):
                for layer2 in range(1, 101):
                    for layer3 in range(1, 101):
                        if layer2 <= layer1 and layer3 <= layer2:
                            param = (layer1, layer2, layer3)
                            test_score, RMSE_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                            if test_score > 0 and RMSE_score < 6:
                                print(param)
            print("layer 3 finish")

"""    for layer1 in range(1, 101):
        if layer1 % 10 == 0:
            print(str(layer1) + " finish at " + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
        for layer2 in range(1, 101):
            for layer3 in range(1, 101):
                if layer2 <= layer1 and layer3 <= layer2:
                    param = (layer1, layer2, layer3)
                    test_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                    if test_score > 0:
                        print(param)
"""

"""    for param_num in range(len(param)):
        if param_num % 10 == 0:
            print(str(param_num) + " finish at " + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
        predict_model, test_score, cross_value = MLP_auto(train_feature, train_target, test_feature, test_target, param[param_num])
        if cross_value < 6:
            print(param[param_num])
            explainer = shap.KernelExplainer(predict_model, feature_train_summary)
            shap_values = explainer.shap_values(train_feature)

            # Dot summary plot
            summary_plot(shap_values, train_feature, plot_type='dot', show=False)
            # plt.colorbar()
            plt.tick_params(labelsize=13)
            plt.title(str(param[param_num]))
            plt.savefig(param_select_path + str(param[param_num]) + ".png")
            # plt.show()
            plt.close()"""
