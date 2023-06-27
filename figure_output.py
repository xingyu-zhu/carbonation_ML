import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, MaxNLocator
from shap import summary_plot
from shap.plots import heatmap, scatter

import shap

SHAP_plot_save_path = './SHAP_plot/'
feature_SHAP_plot_path = 'All feature SHAP plots/'

def get_label_name(feature_name):
    ignore_list = ["Temperature", "Carbonation Time", "Particle Size", "CaO", "MgO", "MnO", "L/S"]
    if feature_name in ignore_list:
        return feature_name
    elif feature_name == "CO2 Partial Pressure":
        return "$CO_2$ CO2 Partial Pressure"
    elif feature_name == "CO2 Concentration":
        return "$CO_2$ Concentration"
    elif feature_name == "SiO2":
        return "$SiO_2"
    elif feature_name == "Al2O3":
        return "$Al_2$$O_3$"
    elif feature_name == "Fe2O3":
        return "$Fe_2$$O_3$"
    else:
        return feature_name

def feature_name_replace(feature_name: str):
    if type(feature_name) is str:
        return feature_name.replace('/', '-')
    else:
        print("\033[32mTypeError: Type of parameter 'feature_name' is str, input value is " + str(type(feature_name))
              + "\033[0m")

def heat_map(data_df):
    heat_map_path = SHAP_plot_save_path + "heatmap/"
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 15
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    data_heat = np.corrcoef(data_df.values, rowvar=0)
    data_heat = pd.DataFrame(data=data_heat, columns=data_df.columns, index=data_df.columns)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(data_heat, square=True, annot=True, fmt='.3f', linewidths=.5, cmap='OrRd',
                     cbar_kws={'fraction': 0.046, 'pad': 0.03})
    plt.savefig(heat_map_path + 'heatmap.png', bbox_inches='tight')

def relevance_plot(train_shap_values, feature_train, feature_name1, feature_name2):
    feature_values1 = feature_train[str(feature_name1)].values
    feature_values2 = feature_train[str(feature_name2)].values
    feature_name1_index = int(feature_train.columns.get_loc(str(feature_name1)))
    feature_name2_index = int(feature_train.columns.get_loc(str(feature_name2)))

    shap_values_sum = train_shap_values[:, feature_name1_index] + train_shap_values[:, feature_name2_index]
    bottom = shap_values_sum.min() - 1
    top = shap_values_sum.max() + 1

    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 10
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1 = plt.axes(projection='3d')
    ax1.set_zlim(bottom, top)
    im = ax1.scatter3D(feature_values1, feature_values2, shap_values_sum, c=shap_values_sum, cmap='jet')
    ax1.scatter3D(feature_values1, feature_values2, bottom - 1)
    ax1.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    plt.grid(True)
    plt.grid(alpha=0.2)
    for number in range(len(shap_values_sum)):
        xs = [feature_values1[number], feature_values1[number]]
        ys = [feature_values2[number], feature_values2[number]]
        zs = [shap_values_sum[number], bottom - 1]
        plt.plot(xs, ys, zs, c='grey', linestyle='--', alpha=0.1, linewidth=0.8)
    plt.tick_params(labelsize=13, pad=0.1)
    plt.xlabel(str(feature_name1), fontsize=15)
    plt.ylabel(str(feature_name2), fontsize=15)
    plt.colorbar(im, fraction=0.1, shrink=0.6, pad=0.1)
    ax1.view_init(elev=20)
    plt.savefig(SHAP_plot_save_path + str(feature_name1) + "_" + str(feature_name2) + ".png")

def material_relevance_plot(feature_train_1, feature_train_2, train_shap_values_1, train_shap_values_2, feature_name1, feature_name2):
    shap_values = train_shap_values_1 + train_shap_values_2
    bottom = int(shap_values.min()) - 1
    top = int(shap_values.max()) + 1

    c = shap_values
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 13
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    ax1 = plt.axes(projection='3d')
    ax1.set_zlim(bottom, top)
    im = ax1.scatter3D(feature_train_1, feature_train_2, shap_values, c=c, cmap='jet')
    ax1.scatter3D(feature_train_1, feature_train_2, -25)
    ax1.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))
    ax1.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.6))

    plt.grid(True)
    plt.grid(alpha=0.2)
    for number in range(len(shap_values)):
        xs = [feature_train_1[number], feature_train_1[number]]
        ys = [feature_train_2[number], feature_train_2[number]]
        zs = [shap_values[number], bottom]
        plt.plot(xs, ys, zs, c='grey', linestyle='--', alpha=0.1, linewidth=0.8)
    plt.tick_params(labelsize=13, pad=0.1)
    plt.xlabel(feature_name1, fontsize=15)
    plt.ylabel(feature_name2, fontsize=15)
    plt.colorbar(im, fraction=0.1, shrink=0.6, pad=0.1)
    ax1.view_init(elev=20)

    plt.savefig(SHAP_plot_save_path + feature_name1 + "_" + feature_name2 + ".png")

def calculate_material(feature_name_1, feature_name_2, train_shap_values, feature_train):
    feature1_num = int(feature_train.columns.get_loc(str(feature_name_1)))
    feature2_num = int(feature_train.columns.get_loc(str(feature_name_2)))
    feature_sum_value = feature_train[str(feature_name_1)].values + feature_train[str(feature_name_2)].values
    shap_sum_value = train_shap_values[:, feature1_num] + train_shap_values[:, feature2_num]

    return feature_sum_value, shap_sum_value

def get_material_shap_value(feature_name, train_shap_value, feature_train):
    feature_num = int(feature_train.columns.get_loc(str(feature_name)))
    feature_value = feature_train[str(feature_name)].values
    feature_shap_value = train_shap_value[:, feature_num]

    return feature_value, feature_shap_value

def SHAP_plot(predict_model, feature_train_summary, feature_train):
    explainer = shap.KernelExplainer(predict_model, feature_train_summary)
    shap_values = explainer.shap_values(feature_train)
    feature_SHAP(9, feature_train['Fe2O3'], shap_values)

    # All features SHAP values plots
    plt.figure(dpi=600, figsize=(10, 8))
    feature_name_list = ["Temperature", "CO2 Partial Pressure", "CO2 Concentration", "Carbonation Time", "Particle Size"
        , "CaO", "MgO", "MnO", "SiO2", "Al2O3", "Fe2O3", "L/S"]

    for feature_name in feature_name_list:
        feature_value, SHAP_value = get_material_shap_value(feature_name, shap_values, feature_train)
        plt.scatter(feature_value, SHAP_value)
        plt.tick_params(labelsize=25, pad=0.1)
        plt.xlabel(get_label_name(feature_name=feature_name) + " Value", fontsize=25)
        plt.ylabel("SHAP Value of " + get_label_name(feature_name=feature_name), fontsize=25)
        plt.savefig(SHAP_plot_save_path + feature_SHAP_plot_path + feature_name_replace(feature_name) + " scatter.png")
        plt.close()

    # Dot summary plot
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 15
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    summary_plot(shap_values, feature_train, plot_type='dot', show=False)
    plt.savefig(SHAP_plot_save_path + "figure.png")
    plt.close()

    # Bar summary plot
    plt.figure(dpi=600, figsize=(6, 6))
    summary_plot(shap_values, feature_train, plot_type='bar', show=False)
    plt.tick_params(labelsize=15)
    plt.savefig(SHAP_plot_save_path + "figure1.png")
    plt.close()

    CaO_MgO_sum_feature_value, CaO_MgO_sum_shap_value = calculate_material("CaO", "MgO", shap_values, feature_train)
    SiO2_Al2O3_feature_value, SiO2_Al2O3_shap_value = calculate_material("SiO2", "Al2O3", shap_values, feature_train)
    Fe2O3_feature_value, Fe2O3_shap_value = get_material_shap_value("Fe2O3", shap_values, feature_train)
    MnO_feature_value, MnO_shap_value = get_material_shap_value("MnO", shap_values, feature_train)

    material_relevance_plot(CaO_MgO_sum_feature_value, SiO2_Al2O3_feature_value, CaO_MgO_sum_shap_value, SiO2_Al2O3_shap_value, "CaO + MgO", "SiO2 + Al2O3")
    plt.close()

    material_relevance_plot(MnO_feature_value, Fe2O3_feature_value, MnO_shap_value, Fe2O3_shap_value, "MnO", "Fe2O3")
    plt.close()

def feature_SHAP(feature_number, feature_train, SHAP_values):
    fig = plt.figure(dpi=600, figsize=(6, 6))
    plt_x = feature_train
    feature_SHAP_value = SHAP_values[:, feature_number]
    plt.scatter(plt_x, feature_SHAP_value)
    plt.savefig(SHAP_plot_save_path + "feature.png")
    plt.close()

def other_SHAP_plot(predict_model, feature_train_summary, feature_train):
    fig = plt.figure(dpi=600, figsize=(6, 6))
    expleiner = shap.Explainer(predict_model, feature_train)
    shap_values = expleiner(feature_train)

    # Heat map
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    heatmap(shap_values, show=False, max_display=12)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(SHAP_plot_save_path + "heatmap.png")
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.plots.scatter(shap_values[:, "CO2 Concentration"], color=shap_values[:, "Temperature"], show=False)
    plt.savefig(SHAP_plot_save_path + "dependence.png")
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.plots.scatter(shap_values[:, "CO2 Concentration"], color=shap_values[:, "CO2 Partial Pressure"], show=False)
    plt.savefig(SHAP_plot_save_path + "dependence1.png")
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.plots.scatter(shap_values[:, "CO2 Concentration"], color=shap_values[:, "Carbonation Time"], show=False)
    plt.savefig(SHAP_plot_save_path + "dependence2.png")
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.plots.scatter(shap_values[:, "CO2 Concentration"], color=shap_values[:, "Particle Size"], show=False)
    plt.savefig(SHAP_plot_save_path + "dependence3.png")
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    shap.plots.scatter(shap_values[:, "CO2 Concentration"], color=shap_values[:, "L/S"], show=False)
    plt.savefig(SHAP_plot_save_path + "dependence4.png")
    plt.close()

def predict_compare_plot(train_target, train_predict, test_target, test_predict):
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 16
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    test_result = pd.concat([pd.DataFrame(test_target.values), pd.DataFrame(test_predict)], axis=1)
    train_result = pd.concat([pd.DataFrame(train_target.values), pd.DataFrame(train_predict)], axis=1)

    plt.plot([0, 60], [0, 60], linestyle='--', alpha=0.7, c='green', label='Baseline')
    plt.plot([0, 50], [0, 60], linestyle='-.', c='r', alpha=0.8, label='20% Offset line')
    plt.plot([0, 60], [0, 48], linestyle='-.', c='r', alpha=0.8)
    plt.scatter(train_result.iloc[:, 0], train_result.iloc[:, 1], marker='o', alpha=0.7, c='w', edgecolors='green', s=60, label='Train Set')
    plt.scatter(test_result.iloc[:, 0], test_result.iloc[:, 1], marker='^', alpha=1, c='w', edgecolors='blue', s=55, label='Test Set')
    plt.tick_params(labelsize=16)
    plt.xlabel('Actual $CO_2$ Sequestration(%)', fontsize='16')
    plt.ylabel('Predicted $CO_2$ Sequestration(%)', fontsize='16')
    plt.legend(loc=2, fontsize=16, markerscale=1, frameon=False)
    x_major_locator = MultipleLocator(10)
    plt.gca().xaxis.set_major_locator(x_major_locator)

    plt.savefig(SHAP_plot_save_path + "predict.png")
    plt.close()

def test_predict_plot(test_predict, test_target):
    plt.figure(dpi=600, figsize=(6, 6))
    plt.rcParams['font.size'] = 16
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    sample_number = []

    for number in range(len(test_predict)):
        sample_number.append(number+1)

    colors = sns.color_palette("colorblind")
    plt.plot(sample_number, test_target, color=colors[1], label="Actual Values", linewidth=2, linestyle='-', marker='o', markersize='6')
    plt.plot(sample_number, test_predict, color=colors[0], label="Predicted Values", linewidth=2, linestyle='--', marker='*', markersize='9')
    plt.xlabel("Sample Number", fontsize=16)
    plt.ylabel("$CO_2$ Sequestration(%)", fontsize=16)
    plt.legend(loc=2, fontsize=16, markerscale=1, frameon=False)
    plt.tick_params(labelsize=16)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim((0, 13))
    plt.ylim((0, 50))
    plt.savefig(SHAP_plot_save_path + "test_predict.png")
    plt.close()
