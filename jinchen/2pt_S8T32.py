# %%

"""
This is a test script for pion 2pt after gauge fixing.

Things to be tested:
1. Using different gauge fixing precisions;
2. Fixing to different Gribov copies;
3. All 16 kinds of gamma matrices and different source positions.

"""

import os

os.sched_setaffinity(0, {0})  # just use one cpu

import gpt as g
import numpy as np

rng = g.random("T")

# Configuration
gamma_idx = 5
gauge = "coulomb"  # "landau" or "coulomb"

gauge_fix = True # True, False, "random"
precision = "1e-8"

fig_name = f"b6_{precision}_coulomb_2pt"
n_conf_ls = np.arange(50)

boundary_phases_t = 1.0

# %%
# Main loop
corr_conf_ls = []

for n_conf in n_conf_ls:
    if gauge_fix == True:
        U_fixed = g.load(f"configs/S8T32/wilson_b6.{gauge}.{precision}.{n_conf}")
    elif gauge_fix == False:
        U_fixed = g.load(f"configs/S8T32/wilson_b6.{n_conf}")
    elif gauge_fix == "random":
        U_ran = g.qcd.gauge.random(g.grid([8, 8, 8, 32], g.double), rng)

        opt = g.algorithms.optimize.non_linear_cg(maxiter=100, eps=1e-9, step=0.1)
        V0 = g.identity(U_ran[0])
        rng.element(V0) #* random initial guess of the gauge transformation

        if gauge == "coulomb":
            c = g.qcd.gauge.fix.coulomb(U_ran)
        elif gauge == "landau":
            c = g.qcd.gauge.fix.landau(U_ran)

        fac = g.algorithms.optimize.fourier_accelerate.inverse_phat_square(V0.grid, c)
        V1 = g.copy(V0)
        opt(fac)([V1], [V1])

        U_fixed = g.qcd.gauge.transformed(U_ran, V1)

    # Quark and solver setup (same for all source positions)
    grid = U_fixed[0].grid
    L = np.array(grid.fdimensions)

    w = g.qcd.fermion.wilson_clover(
        U_fixed,
        {
            "kappa": 0.137,
            "csw_r": 0,
            "csw_t": 0,
            "xi_0": 1,
            "nu": 1,
            "isAnisotropic": False,
            "boundary_phases": [1.0, 1.0, 1.0, boundary_phases_t],
        },
    )
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-6, "maxiter": 1000})
    propagator = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))

    # create point source
    src = g.mspincolor(grid)
    g.create.point(src, [0, 0, 0, 0])

    # propagator
    dst = g.mspincolor(grid)
    dst @= propagator * src

    # momentum
    p = 2.0 * np.pi * np.array([1, 0, 0, 0]) / L
    P = g.exp_ixp(p)

    # operators
    G_src = g.gamma[gamma_idx] * P
    G_snk = g.gamma[gamma_idx] * g.adj(P)

    # 2pt
    correlator_2pt = g.slice(g.trace(G_src * g.gamma[5] * g.adj(dst) * g.gamma[5] * G_snk * dst), 3)

    corr_conf_ls.append(np.real(correlator_2pt))


# %%
import gvar as gv
import matplotlib.pyplot as plt
from func.resampling import bootstrap, jackknife, bs_ls_avg, jk_ls_avg
from func.plot_settings import *

# Data analysis
# pt2_conf_array = bootstrap(corr_conf_ls, 100)
pt2_conf_array = jackknife(corr_conf_ls)
pt2_conf_avg = jk_ls_avg(pt2_conf_array)

# * Effective mass no boundary effects
# meff = np.log(pt2_conf_avg[:-1] / pt2_conf_avg[1:])

# * Effective mass with boundary effects
# ArcSinh [(Cx[t + 1] + Cx[t - 1])/(2Cx[t])]

if boundary_phases_t == -1.0:
    meff = np.arcsinh((pt2_conf_avg[:-2] + pt2_conf_avg[2:]) / (2 * pt2_conf_avg[1:-1]))
elif boundary_phases_t == 1.0:
    meff = np.arccosh((pt2_conf_avg[2:] + pt2_conf_avg[:-2]) / (2 * pt2_conf_avg[1:-1]))


# Plotting
def plot_data(data, title, filename):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.errorbar(
        range(len(data)), gv.mean(data), yerr=gv.sdev(data), fmt=".", color="black"
    )
    ax.set_xlabel("t")
    ax.set_title(title)
    if title == "correlator":
        ax.set_yscale("log")
    elif title == "effective mass":
        ax.set_ylim(-5, 5)
    plt.savefig(filename)
    plt.show()


plot_data(meff, "effective mass", f"meff_S8T32_{fig_name}.png")
plot_data(pt2_conf_avg, "correlator", f"corr_S8T32_{fig_name}.png")


# %%
#! test

print(pt2_conf_avg)
print(meff)

gv.dump(meff, f"dump/pion_meff_{gauge}_{precision}.dat")


# %%
def plot_data_double(data1, data2, title, filename):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.errorbar(
        range(len(data1)), gv.mean(data1), yerr=gv.sdev(data1), fmt=".", color="black"
    )
    ax.errorbar(
        range(len(data2)), gv.mean(data2), yerr=gv.sdev(data2), fmt=".", color="red"
    )
    ax.set_xlabel("t")
    ax.set_title(title)
    if title == "correlator":
        ax.set_yscale("log")
    elif title == "effective mass":
        ax.set_ylim(0.5, 2.5)
    plt.savefig(filename)
    plt.show()

data1 = gv.load("dump/pion_meff_coulomb_1e-8.dat")
data2 = gv.load("dump/pion_meff_coulomb_1e-6.dat")

plot_data_double(data1, data2, "effective mass", "meff_S8T32_coulomb_comparison.png")
# %%
