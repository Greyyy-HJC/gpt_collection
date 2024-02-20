'''
This is a test script to generate a set of gauge configurations after being heat to be balanced, then save them to a file.

Two parts:
1. Markov chain to generate the configurations;
2. Save the configurations to a file.
'''

# %%
import os
os.sched_setaffinity(0, {0})  # just use one cpu

import gpt as g

# grid
L = [8, 8, 8, 32]
grid = g.grid(L, g.double)
grid_eo = g.grid(L, g.double, g.redblack)

# hot start
g.default.push_verbose("random", False)
rng = g.random("test", "vectorized_ranlux24_24_64")
U = g.qcd.gauge.unit(grid)
Nd = len(U)

# red/black mask
mask_rb = g.complex(grid_eo)
mask_rb[:] = 1

# full mask
mask = g.complex(grid)

# action
w = g.qcd.gauge.action.wilson(6.0)

# heatbath sweeps
g.default.push_verbose("su2_heat_bath", False)
markov = g.algorithms.markov.su2_heat_bath(rng)
U = g.qcd.gauge.unit(grid)


# %%
#! heat balance
if True:
    for it in range(50):
        plaq = g.qcd.gauge.plaquette(U)
        R_2x1 = g.qcd.gauge.rectangle(U, 2, 1)
        g.message(f"SU(2)-subgroup heatbath {it} has P = {plaq}, R_2x1 = {R_2x1}")
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)

            for mu in range(Nd):
                markov(U[mu], w.staple(U, mu), mask)

# %%
if True:
    g.save("configs/S8T32/balance_S8T32_b6", U)
    U_check = g.load("configs/S8T32/balance_S8T32_b6")

    g.message( g.norm2(U_check[1] - U[1]) )


# %%
#! save configs
U_it = g.load("configs/S8T32/balance_S8T32_b6")

for n_conf in range(50):
    for gap in range(40):
        it = n_conf * 40 + gap

        plaq = g.qcd.gauge.plaquette(U_it)
        R_2x1 = g.qcd.gauge.rectangle(U_it, 2, 1)
        g.message(f"SU(2)-subgroup heatbath {it} has P = {plaq}, R_2x1 = {R_2x1}")
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)

            for mu in range(Nd):
                markov(U_it[mu], w.staple(U_it, mu), mask)

    g.save(f"configs/S8T32/wilson_b6.{n_conf}", U_it)

# %%

