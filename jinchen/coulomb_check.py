"""
added by Jinchen on 2023.12.14

The only difference between Coulomb gauge and Landau gauge is the functional form, here we just sum over the spatial components, i.e. [:-1] in the return of __call__ function.
"""

# %%
import os

os.sched_setaffinity(0, {0})  # just use one cpu

import gpt as g
import numpy as np
from gpt.core.group import differentiable_functional

rng = g.random("T")

class coulomb(differentiable_functional):
    def __init__(self, U):
        self.U = U

    def __call__(self, V):
        V = g.util.from_list(V)
        functional = 0.0
        for mu in range(3): #todo  # Only spatial directions
            transformed_links = g.qcd.gauge.transformed(self.U, V)[mu]
            functional += g.sum(g.trace(transformed_links))
        return functional.real * (-2.0)

    @differentiable_functional.single_field_gradient
    def gradient(self, V):
        A = [
            g(g.qcd.gauge.project.traceless_anti_hermitian(u) / 1j)
            for u in g.qcd.gauge.transformed(self.U, V)
        ]
        dmuAmu = V.new()
        dmuAmu.otype = V.otype.cartesian()
        dmuAmu = g(0.0 * dmuAmu)
        for mu in range(3):  #todo # Only spatial directions
            Amu = A[mu]
            dmuAmu += Amu - g.cshift(Amu, mu, -1)
        return dmuAmu


def fix_gauge(U, eps, step):
    """
    Input a gauge field U, output the gauge field after gauge fixing
    """
    V0 = g.identity(U[1])
    rng.element(V0)

    c = coulomb(U)

    fac = g.algorithms.optimize.fourier_accelerate.inverse_phat_square(V0.grid, c)

    V1 = g.copy(V0)

    opt = g.algorithms.optimize
    cg = opt.non_linear_cg(maxiter=1000, eps=eps, step=step)  
    cg(fac)([V1], [V1])

    U_fixed = g.qcd.gauge.transformed(U, V1)

    return U_fixed


#! check gauge fixing
if True:
    eps = 1e-6
    step = 1e-2

    U_read = g.convert( g.load('configs/gauge_disord_4c8.NERSC'), g.double )

    for i in range(4):
        g.message('>>> i = ', i)
        g.message( g.norm2(U_read[i]) / (3 * 4 * 4 * 4 * 8) )
        g.message( np.real( g.sum(g.trace(U_read[i])) ) / (3 * 4 * 4 * 4 * 8) )

    U_fixed = fix_gauge(U_read, eps, step)

    for i in range(4):
        g.message('>>> i = ', i)
        g.message( g.norm2(U_fixed[i]) / (3 * 4 * 4 * 4 * 8) )
        g.message( np.real( g.sum(g.trace(U_fixed[i])) ) / (3 * 4 * 4 * 4 * 8) )

# %%
