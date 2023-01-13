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

param = [(9, 4, 4),(10, 7, 6), (10, 8, 2), (11, 6, 6), (11, 11, 11),(13, 12, 11), (14, 8, 1), (15, 4, 3), (15, 12, 11),
         (17, 1, 1), (18, 14, 5), (19, 19, 13), (20, 7, 4), (20, 11, 2), (20, 15, 9), (21, 3, 2),
         (21, 12, 7), (21, 13, 5), (21, 18, 16), (23, 15, 8), (24, 16, 12), (25, 21, 9), (26, 23, 5), (26, 26, 10), (28, 4, 4),
         (28, 20, 5), (29, 3, 2), (29, 10, 4), (29, 18, 4), (29, 24, 5), (29, 26, 4), (29, 27, 9), (31, 14, 3), (31, 14, 4), (32, 6, 5),
         (32, 7, 5), (32, 14, 1), (32, 19, 19), (33, 18, 4), (33, 22, 12), (34, 9, 2), (35, 27, 18), (36, 14, 8), (36, 30, 26),
         (37, 10, 1), (37, 10, 2), (38, 7, 7), (38, 33, 3), (39, 19, 10), (39, 20, 4), (39, 21, 2), (39, 29, 4), (39, 31, 3),
         (40, 21, 10), (40, 22, 6), (40, 26, 7), (40, 39, 8), (41, 7, 4), (42, 4, 3), (42, 18, 3), (42, 22, 22), (42, 29, 14),
         (42, 32, 11), (42, 39, 37), (43, 12, 9), (43, 14, 13), (43, 26, 5), (43, 27, 6), (43, 31, 9), (43, 36, 26), (43, 38, 37),
         (44, 17, 13), (44, 20, 8), (44, 30, 3),(44, 39, 10), (44, 40, 33), (46, 42, 13), (46, 42, 36), (46, 45, 29), (47, 29, 5),
         (47, 29, 17), (48, 7, 4), (48, 9, 5), (48, 12, 2), (48, 22, 4), (49, 4, 4), (49, 16, 6), (49, 46, 16), (49, 47, 8), (50, 14, 7),
         (50, 30, 26), (51, 14, 6), (51, 26, 12), (51, 37, 5), (51, 46, 25), (51, 47, 30), (51, 49, 4), (51, 49, 37), (52, 39, 10),
         (52, 43, 8), (52, 45, 7), (52, 52, 36), (53, 25, 16), (53, 33, 3), (53, 44, 12), (53, 53, 10), (54, 1, 1), (54, 13, 3),
         (55, 46, 3), (55, 48, 36), (56, 10, 7), (56, 17, 13), (56, 24, 8), (56, 30, 3), (56, 39, 22), (56, 42, 25), (56, 48, 10),
         (56, 49, 48), (57, 22, 10), (57, 29, 5), (57, 29, 19), (57, 38, 15), (57, 50, 18), (58, 24, 6), (58, 47, 37), (58, 52, 37),
         (59, 9, 5), (59, 22, 6), (59, 50, 11), (59, 58, 27), (60, 50, 5), (60, 50, 28), (61, 5, 3), (61, 11, 6), (61, 24, 7)]

def run_auto_param(train_feature, train_target, test_feature, test_target, param, feature_train_summary):
    param_select_path = './select_param/'
    score = 0
    best_param = 0

    for layer1 in range(1, 101):
        if layer1 % 10 == 0:
            print(str(layer1) + " finish at " + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
        for layer2 in range(1, 101):
            for layer3 in range(1, 101):
                if layer2 <= layer1 and layer3 <= layer2:
                    param = (layer1, layer2, layer3)
                    test_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                    if test_score > 0:
                        if test_score > score:
                            score = test_score
                            best_param = param
                        print(param)


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


"""def run_auto_param(train_feature, train_target, test_feature, test_target):
    score = 0
    best_param = 0
    for layer_number in range(1, 4):
        if layer_number == 1:
            for layer1 in range(1, 101):
                param = (layer1)
                test_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                if test_score > 0:
                    if test_score > score:
                        score = test_score
                        best_param = param
                    print(param)
            print("layer 1 finish")
        elif layer_number == 2:
            for layer1 in range(1, 101):
                for layer2 in range(1, 101):
                    if layer2 <= layer1:
                        param = (layer1, layer2)
                        test_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                        if test_score > 0:
                            if test_score > score:
                                score = test_score
                                best_param = param
                            print(param)
            print("layer 2 finish")
        elif layer_number == 3:
            for layer1 in range(1, 101):
                for layer2 in range(1, 101):
                    for layer3 in range(1, 101):
                        if layer2 <= layer1 and layer3 <= layer2:
                            param = (layer1, layer2, layer3)
                            test_score = MLP_auto(train_feature, train_target, test_feature, test_target, param)
                            if test_score > 0:
                                if test_score > score:
                                    score = test_score
                                    best_param = param
                                print(param)
            print("layer 3 finish")

    print(best_param)"""
