import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

bar_plot_savepath = './SHAP_plot/'

def bar_plot(all_feature_value):

    # plt.rcParams["font.weight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    # sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["CO2 Concentration"], kde=True, alpha=1, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'CO2 Concentration_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["Temperature"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'Temperature_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["CO2 Partial Pressure"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'CO2 Partial Pressure_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["Carbonation Time"], kde=True, alpha=0.7, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'Carbonation Time_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["Particle Size"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'Particle Diameter_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["CaO"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'CaO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["MgO"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'MgO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["SiO2"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'SiO2_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["Al2O3"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'Al2O3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["Fe2O3"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'Fe2O3_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["MnO"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'MnO_statistics.png')
    plt.close()

    fig = plt.figure(dpi=600, figsize=(6, 6))
    sns.set_palette("hls")
    mpl.rc("figure", figsize=(10, 10))
    sns.set(context='paper', style='ticks', font_scale=2)
    sns.displot(all_feature_value["L/S"], kde=True, alpha=0.3, color='r', height=5, aspect=1.5)
    plt.savefig(bar_plot_savepath + 'LS_statistics.png')
    plt.close()
