# %%
import os

os.sched_setaffinity(0, {0})  # just use one cpu

import gpt as g
import numpy as np

rng = g.random("T")

size = "S8T32"
gauge_todo = "coulomb"
precision = 1e-9
precision_aim = "1e-9"
n_conf_ls = np.arange(50)
save_gauge_fix = True
redo = True
precision_base = "1e-9"


def fix_gauge(U, gauge):
    """
    Input a gauge field U, output the gauge field after gauge fixing
    """
    V0 = g.identity(U[1])
    rng.element(V0)

    if gauge == "coulomb":
        c = g.qcd.gauge.fix.coulomb(U)

    elif gauge == "landau":
        c = g.qcd.gauge.fix.landau(U)

    fac = g.algorithms.optimize.fourier_accelerate.inverse_phat_square(V0.grid, c)

    V1 = g.copy(V0)

    
    # eps = 1e-8
    # step = 1e-11
    opt = g.algorithms.optimize
    cg = opt.non_linear_cg(maxiter=1000, eps=precision, step=1e-13)  # todo
    gd = opt.gradient_descent(maxiter=10000, eps=precision, step=0.01)
    if not cg(fac)([V1], [V1]):
        gd(fac)([V1], [V1])

    # while eps > (precision / np.sqrt(10)):
    #     cg = opt.non_linear_cg(
    #         maxiter=200, eps=eps, step=step
    #     )  # todo
    #     gd = opt.gradient_descent(maxiter=600, eps=eps, step=0.03)

    #     if not cg(fac)([V1], [V1]):
    #         gd(fac)([V1], [V1])

    #     eps /= np.sqrt(10)
    #     step /= np.sqrt(10)

    U_fixed = g.qcd.gauge.transformed(U, V1)

    return U_fixed


#! gauge fixing
if True:
    for n_conf in n_conf_ls:
        if redo == True:
            U_read = g.load(f"configs/{size}/wilson_b6.{gauge_todo}.{precision_base}.{n_conf}")
        elif redo == False:
            U_read = g.load(f"configs/{size}/wilson_b6.{n_conf}")

        U_fixed = fix_gauge(U_read, gauge=gauge_todo)

        if save_gauge_fix:
            g.save(f"configs/{size}/wilson_b6.{gauge_todo}.{precision_aim}.{n_conf}", U_fixed)


#! check gauge fixing
if False:
    U_read = g.convert( g.load('configs/gauge_disord_4c8.NERSC'), g.double )

    for i in range(4):
        g.message('>>> i = ', i)
        g.message( g.norm2(U_read[i]) / (3 * 4 * 4 * 4 * 8) )
        g.message( np.real( g.sum(g.trace(U_read[i])) ) / (3 * 4 * 4 * 4 * 8) )

    # U_read = g.qcd.gauge.random(g.grid([4, 4, 4, 8], g.double), rng)
    U_fixed = fix_gauge(U_read, 'coulomb')

    for i in range(4):
        g.message('>>> i = ', i)
        g.message( g.norm2(U_fixed[i]) / (3 * 4 * 4 * 4 * 8) )
        g.message( np.real( g.sum(g.trace(U_fixed[i])) ) / (3 * 4 * 4 * 4 * 8) )

    # g.save('configs/gauge_disord_4c8.coulomb', U_fixed, g.format.nersc())
# %%
