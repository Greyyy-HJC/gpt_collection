# %%

"""
This is a test script for quark propagator after gauge fixing. Also keep the source-sink separation to be the same.

Things to be tested:
1. The gauge fixing is successful, i.e. the quark prop does not vanish;
2. Using different gauge fixing precisions;
3. Fixing to different Gribov copies;
4. All 16 kinds of gamma matrices and different source positions.

"""

import os

os.sched_setaffinity(0, {1})  # just use one cpu

import gpt as g
import numpy as np

rng = g.random("T")

# Configuration
gamma_idx = "I"
gauge = "coulomb"  # "landau" or "coulomb"

gauge_fix = True # True, False, "random"
precision = "1e-9"

fig_name = f"b6_{gauge}_periodic_0000"
n_conf_ls = np.arange(50)

boundary_phases_t = 1.0
src_positions = [(x, y, z, 0) for x in range(1) for y in range(1) for z in range(1)]
sep = 0  # source-sink separation

take_even_t = False

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

    # momentum
    p = 2.0 * np.pi * np.array([1, 0, 0, 0]) / L
    P = g.exp_ixp(p)

    # Source positions
    for x, y, z, t in src_positions:
        src = g.mspincolor(grid)
        g.create.point(src, [x, y, z, t])
        dst = g.mspincolor(grid)
        dst @= propagator * src
        correlator = g(g.trace(dst * g.gamma[gamma_idx]))[x, y, z + sep, :].flatten()

        #todo: z decay
        # correlator = g(g.trace(dst * g.gamma[gamma_idx]))[x, y, :, 0].flatten()

        #todo: two prop to get 2pt
        # correlator = g(g.trace(dst * g.gamma[gamma_idx] * g.adj(dst)))[x, y, z + sep, :].flatten()

        corr_conf_ls.append(np.real(correlator))

print(corr_conf_ls)

# %%
import gvar as gv
import matplotlib.pyplot as plt
from func.resampling import bootstrap, jackknife, bs_ls_avg, jk_ls_avg
from func.plot_settings import *

# take_even_t = False

# Data analysis
# pt2_conf_array = bootstrap(corr_conf_ls, 100)
# pt2_conf_avg = bs_ls_avg(pt2_conf_array)

pt2_conf_array = jackknife(corr_conf_ls)
pt2_conf_avg = jk_ls_avg(pt2_conf_array)
if take_even_t:
    pt2_conf_avg = pt2_conf_avg[::2]


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
# gv.dump(meff, f"dump/qk_meff_{gauge}_{precision}.dat")
# gv.dump(pt2_conf_avg, f"dump/qk_prop_{gauge}_{precision}.dat")

# %%
