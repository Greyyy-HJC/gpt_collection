# %%
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from func.plot_settings import *

# Plotting
def plot_data(data_ls, label_ls, title, filename):
    fig, ax = plt.subplots(figsize=fig_size)
    i = 0
    for (data, label) in zip(data_ls, label_ls):
        ax.errorbar(
            np.arange(len(data))+i*0.05, gv.mean(data), yerr=gv.sdev(data), fmt=marker_ls[i], color=color_ls[i], label=label
        )
        i+=1
    ax.set_xlabel("t")
    ax.set_title(title)
    if title == "correlator":
        ax.set_yscale("log")
        # ax.set_ylim(-1e-5, 1e-5)
    elif title == "effective mass":
        ax.set_ylim(-5, 5)
    plt.legend()
    if filename != None:
        plt.savefig(filename)
    plt.show()


# %%
#! Coulomb gauge
#* seems no signal

qk_prop_coulomb_1e4 = gv.load('dump/qk_prop_coulomb_1e-4.dat')
qk_prop_coulomb_1e6 = gv.load('dump/qk_prop_coulomb_1e-6.dat')
qk_prop_coulomb_1e8 = gv.load('dump/qk_prop_coulomb_1e-8.dat')
qk_prop_coulomb_1e9 = gv.load('dump/qk_prop_coulomb_1e-9.dat')

qk_meff_coulomb_1e4 = gv.load('dump/qk_meff_coulomb_1e-4.dat')
qk_meff_coulomb_1e6 = gv.load('dump/qk_meff_coulomb_1e-6.dat')
qk_meff_coulomb_1e8 = gv.load('dump/qk_meff_coulomb_1e-8.dat')
qk_meff_coulomb_1e9 = gv.load('dump/qk_meff_coulomb_1e-9.dat')

label_ls = ['CL 1e-4', 'CL 1e-6', 'CL 1e-8', 'CL 1e-9']

plot_data([qk_prop_coulomb_1e4, qk_prop_coulomb_1e6, qk_prop_coulomb_1e8, qk_prop_coulomb_1e9], label_ls, "correlator", 'figs/qk_corr_S8T32_coulomb_comparison.png')
plot_data([qk_meff_coulomb_1e4, qk_meff_coulomb_1e6, qk_meff_coulomb_1e8, qk_meff_coulomb_1e9], label_ls, "effective mass", 'figs/qk_meff_S8T32_coulomb_comparison.png')

# %%
#! Landau gauge

qk_meff_landau_1e4 = gv.load('dump/qk_meff_landau_1e-4.dat')
qk_meff_landau_1e6 = gv.load('dump/qk_meff_landau_1e-6.dat')
qk_meff_landau_1e8 = gv.load('dump/qk_meff_landau_1e-8.dat')

label_ls = ['LD 1e-4', 'LD 1e-6', 'LD 1e-8']

plot_data([qk_meff_landau_1e4, qk_meff_landau_1e6, qk_meff_landau_1e8], label_ls, "effective mass", 'figs/qk_meff_S8T32_landau_comparison.png')


# %%
